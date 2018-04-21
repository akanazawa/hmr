""" Convert Coco to TFRecords """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs

import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO

from .common import convert_to_example, ImageCoder, resize_img

tf.app.flags.DEFINE_string('data_directory', '/scratch1/storage/coco/',
                           'data directory: top of coco')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch1/projects/tf_datasets/coco_wmask/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 500,
                            'Number of shards in validation TFRecord files.')
FLAGS = tf.app.flags.FLAGS

joint_names = [
    'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
    'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
    'Head', 'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear'
]


def convert_coco2universal(kp):
    """
    Mapping from COCO joints (kp: 17 x 3) to
    Universal 19 joints (14 lsp)+ (5 coco faces).

    Permutes and adds extra 0 two rows for missing head and neck
    returns: 19 x 3
    """

    UNIVERSAL_BODIES = [
        16,  # R ankle
        14,  # R knee
        12,  # R hip
        11,  # L hip
        13,  # L knee
        15,  # L ankle
        10,  # R Wrist
        8,  # R Elbow
        6,  # R shoulder
        5,  # L shoulder
        7,  # L Elbow
        9,  # L Wrist
    ]
    UNIVERSAL_HEADS = range(5)
    new_kp = np.vstack((kp[UNIVERSAL_BODIES, :], np.zeros((2, 3)),
                        kp[UNIVERSAL_HEADS, :]))
    return new_kp


def get_anns_details(anns, coco, min_vis=5, min_max_height=60):
    """
    anns is the list of annotations
    coco is the cocoAPI

    extracts the boundingbox (using the mask)
    and the keypoints for each person.

    Ignores the person if there is no or < min_vis keypoints
    Ignores the person if max bbox length is <= min_max_height
    """
    points_other_than_faceshoulder = [
        16,  # R ankle
        14,  # R knee
        12,  # R hip
        11,  # L hip
        13,  # L knee
        15,  # L ankle
        10,  # R Wrist
        8,  # R Elbow
        7,  # L Elbow
        9,  # L Wrist
    ]
    filtered_anns = []
    kps = []
    centers, bboxes = [], []
    masks = []
    for ann in anns:
        if 'keypoints' not in ann or type(ann['keypoints']) != list:
            # Ignore those without keypoints
            continue
        if ann['num_keypoints'] == 0:
            continue

        if 'segmentation' in ann:
            # Use the mask to compute center
            mask = coco.annToMask(ann)
            # import ipdb; ipdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.ion()
            # plt.figure(1)
            # plt.imshow(mask)
            # plt.pause(1e-3)
            # this is N x 2 (in [x, y]) of fgpts
            fg_pts = np.transpose(np.nonzero(mask))[:, ::-1]
            min_pt = np.min(fg_pts, axis=0)
            max_pt = np.max(fg_pts, axis=0)
            bbox = [min_pt, max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
            center = (min_pt + max_pt) / 2.
        else:
            print('No segmentation!')
            import ipdb
            ipdb.set_trace()

        kp_raw = np.array(ann['keypoints'])
        x = kp_raw[0::3]
        y = kp_raw[1::3]
        v = kp_raw[2::3]
        # At least min_vis many visible (not occluded) kps.
        if sum(v == 2) >= min_vis and max(bbox[2:]) > min_max_height:
            # If only face & shoulder visible, skip.
            if np.all(v[points_other_than_faceshoulder] == 0):
                continue
            kp = np.vstack([x, y, v]).T
            kps.append(kp)
            filtered_anns.append(ann)
            centers.append(center)
            bboxes.append(bbox)
            masks.append(mask)

    return filtered_anns, kps, bboxes, centers, masks


def parse_people(kps, centers, masks):
    '''
    Parses people i.e. figures out scale from annotation.
    Input:

    Returns:
      people - list of tuple (kp, img_scale, obj_pos) in this image.
    '''
    # No single persons in this image.
    if len(kps) == 0:
        return []

    # Read each human:
    people = []

    for kp, center, mask in zip(kps, centers, masks):
        # Universal joints!
        joints = convert_coco2universal(kp).T
        # Scale person to be roughly 150x height
        visible = joints[2, :].astype(bool)
        min_pt = np.min(joints[:2, visible], axis=1)
        max_pt = np.max(joints[:2, visible], axis=1)
        person_height = np.linalg.norm(max_pt - min_pt)

        R_ank = joint_names.index('R Ankle')
        L_ank = joint_names.index('L Ankle')

        # If ankles are visible
        if visible[R_ank] or visible[L_ank]:
            my_scale = 150. / person_height
        else:
            L_should = joint_names.index('L Shoulder')
            L_hip = joint_names.index('L Hip')
            R_should = joint_names.index('R Shoulder')
            R_hip = joint_names.index('R Hip')
            # Torso points left should, right shold, right hip, left hip
            # torso_points = joints[:, [9, 8, 2, 3]]
            torso_heights = []
            if visible[L_should] and visible[L_hip]:
                torso_heights.append(
                    np.linalg.norm(joints[:2, L_should] - joints[:2, L_hip]))
            if visible[R_should] and visible[R_hip]:
                torso_heights.append(
                    np.linalg.norm(joints[:2, R_should] - joints[:2, R_hip]))
            # Make torso 75px
            if len(torso_heights) > 0:
                my_scale = 75. / np.mean(torso_heights)
            else:  # No torso!
                body_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11])
                if np.all(visible[body_inds] == 0):
                    print('Face only! skip..')
                    continue
                else:
                    my_scale = 50. / person_height

        people.append((joints, my_scale, center, mask))

    return people


def add_to_tfrecord(coco, img_id, img_dir, coder, writer, is_train):
    """
    Add each "single person" in this image.
    coco - coco API

    Returns:
      The number of people added.
    """
    # Get annotation id for this guy
    # Cat ids is [1] for human..
    ann_id = coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
    anns = coco.loadAnns(ann_id)
    # coco.showAnns(anns)
    filtered_anns, kps, bboxes, centers, masks = get_anns_details(
        anns, coco, min_vis=6, min_max_height=60)

    # Figure out the scale and pack each one in a tuple
    people = parse_people(kps, centers, masks)

    if len(people) == 0:
        # print('No single persons in img %d' % img_id)
        return 0

    # Add each people to tf record
    img_data = coco.loadImgs(img_id)[0]
    image_path = join(img_dir, img_data['file_name'])
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()

    image = coder.decode_jpeg(image_data)

    for joints, scale, pos, mask in people:
        # Scale image:
        image_scaled, scale_factors = resize_img(image, scale)
        height, width = image_scaled.shape[:2]
        joints_scaled = np.copy(joints)
        joints_scaled[0, :] *= scale_factors[0]
        joints_scaled[1, :] *= scale_factors[1]
        # center = pos * scale_factors

        visible = joints[2, :].astype(bool)
        min_pt = np.min(joints_scaled[:2, visible], axis=1)
        max_pt = np.max(joints_scaled[:2, visible], axis=1)
        center = (min_pt + max_pt) / 2.

        ## Crop 400x400 around this image..
        margin = 200
        start_pt = np.maximum(center - margin, 0).astype(int)
        end_pt = (center + margin).astype(int)
        end_pt[0] = min(end_pt[0], width)
        end_pt[1] = min(end_pt[1], height)
        image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
            0], :]
        # Update others oo.
        joints_scaled[0, :] -= start_pt[0]
        joints_scaled[1, :] -= start_pt[1]
        center -= start_pt
        height, width = image_scaled.shape[:2]

        # Vis:
        """
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        image_with_skel = draw_skeleton(image, joints[:2, :], vis=visible, radius=(np.mean(image.shape[:2]) * 0.01).astype(int))
        ax.imshow(image_with_skel)
        ax.axis('off')
        # ax.imshow(image)
        # ax.scatter(joints[0, visible], joints[1, visible])
        # ax.scatter(joints[0, ~visible], joints[1, ~visible], color='green')
        ax.scatter(pos[0], pos[1], color='red')
        ax = fig.add_subplot(122)
        image_with_skel_scaled = draw_skeleton(image_scaled, joints_scaled[:2, :], vis=visible, radius=max(4, (np.mean(image_scaled.shape[:2]) * 0.01).astype(int)))
        ax.imshow(image_with_skel_scaled)
        ax.scatter(center[0], center[1], color='red')
        # ax.imshow(image_scaled)
        # ax.scatter(joints_scaled[0, visible], joints_scaled[1, visible])
        # ax.scatter(pos_scaled[0], pos_scaled[1], color='red')
        ax.axis('on')
        plt.draw()
        plt.pause(0.01)
        """

        # Encode image:
        image_data_scaled = coder.encode_jpeg(image_scaled)
        example = convert_to_example(image_data_scaled, image_path, height,
                                     width, joints_scaled, center)
        writer.write(example.SerializeToString())

    # Finally return how many were written.
    return len(people)


def process_coco(data_dir, out_dir, num_shards, is_train=True):

    if is_train:
        data_type = 'train2014'
        out_path = join(out_dir, 'train_%04d_wmeta.tfrecord')
    else:
        data_type = 'val2014'
        out_path = join(out_dir, 'val_%04d_wmeta.tfrecord')

    anno_file = join(data_dir,
                     'annotations/person_keypoints_%s.json' % data_type)
    img_dir = join(data_dir, 'images', data_type)
    # initialize COCO api for person keypoints annotations
    coco = COCO(anno_file)
    catIds = coco.getCatIds(catNms=['person'])
    img_inds = coco.getImgIds(catIds=catIds)
    # Only run on  'single person's
    coder = ImageCoder()

    i = 0
    # Count on shards
    fidx = 0
    num_ppl = 0
    total_num_ppl = 0
    while i < len(img_inds):
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while i < len(img_inds) and num_ppl < num_shards:
                if i % 100 == 0:
                    print('Reading img %d/%d' % (i, len(img_inds)))
                num_ppl += add_to_tfrecord(coco, img_inds[i], img_dir, coder,
                                           writer, is_train)
                i += 1
            total_num_ppl += num_ppl

        fidx += 1

    print('Made %d shards, with total # of people: %d' %
          (fidx - 1, total_num_ppl))


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)
    process_coco(
        FLAGS.data_directory,
        FLAGS.output_directory,
        FLAGS.train_shards,
        is_train=True)
    # do_valid
    # _process_coco(FLAGS.data_directory, FLAGS.output_directory, FLAGS.validation_shards, is_train=False)


if __name__ == '__main__':
    tf.app.run()
