"""
Convert MPII to TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs
from os.path import join, exists
from time import time

import numpy as np

import tensorflow as tf

from .common import convert_to_example, ImageCoder, resize_img

tf.app.flags.DEFINE_string('img_directory',
                           '/scratch1/storage/human_datasets/mpii',
                           'image data directory')
tf.app.flags.DEFINE_string(
    'output_directory', '/Users/kanazawa/projects/datasets/tf_datasets/mpii',
    'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 500,
                            'Number of shards in validation TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def load_anno(fname):
    import scipy.io as sio
    t0 = time()
    print('Reading annotation..')
    res = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    print('took %g sec..' % (time() - t0))

    return res['RELEASE']


def convert_is_visible(is_visible):
    """
    this field is u'1' or empty numpy array..
    """
    if isinstance(is_visible, np.ndarray):
        assert (is_visible.size == 0)
        return 0
    else:
        return int(is_visible)


def read_joints(rect):
    """
    Reads joints in the common joint order.
    Assumes rect has annopoints as field.

    Returns:
      joints: 3 x |common joints|
    """
    # Mapping from MPII joints to LSP joints (0:13). In this roder:
    _COMMON_JOINT_IDS = [
        0,  # R ankle
        1,  # R knee
        2,  # R hip
        3,  # L hip
        4,  # L knee
        5,  # L ankle
        10,  # R Wrist
        11,  # R Elbow
        12,  # R shoulder
        13,  # L shoulder
        14,  # L Elbow
        15,  # L Wrist
        8,  # Neck top
        9,  # Head top
    ]
    assert ('annopoints' in rect._fieldnames)
    points = rect.annopoints.point
    if not isinstance(points, np.ndarray):
        # There is only one! so ignore this image
        return None
    # Not all joints are there.. read points in a dict.
    read_points = {}

    for point in points:
        vis = convert_is_visible(point.is_visible)
        read_points[point.id] = np.array([point.x, point.y, vis])

    # Go over each common joint ids
    joints = np.zeros((3, len(_COMMON_JOINT_IDS)))
    for i, jid in enumerate(_COMMON_JOINT_IDS):
        if jid in read_points.keys():
            joints[:, i] = read_points[jid]
            # If it's annotated, then use it as visible
            # (in this visible = 0 iff no gt label)
            joints[2, i] = 1.

    return joints


def parse_people(anno_info, single_persons):
    '''
    Parses people from rect annotation.
    Assumes input is train data.
    Input:
      img_dir: str
      anno_info: annolist[img_id] obj
      single_persons: rect id idx for "single" people

    Returns:
      people - list of annotated single-people in this image.
      Its Entries are tuple (label, img_scale, obj_pos)
    '''
    # No single persons in this image.
    if single_persons.size == 0:
        return []

    rects = anno_info.annorect
    if not isinstance(rects, np.ndarray):
        rects = np.array([rects])

    # Read each human:
    people = []

    for ridx in single_persons:
        rect = rects[ridx - 1]
        pos = np.array([rect.objpos.x, rect.objpos.y])
        joints = read_joints(rect)
        if joints is None:
            continue
        # Compute the scale using the keypoints so the person is 150px.
        visible = joints[2, :].astype(bool)
        # If ankles are visible
        if visible[0] or visible[5]:
            min_pt = np.min(joints[:2, visible], axis=1)
            max_pt = np.max(joints[:2, visible], axis=1)
            person_height = np.linalg.norm(max_pt - min_pt)
            scale = 150. / person_height
        else:
            # Torso points left should, right shold, right hip, left hip
            # torso_points = joints[:, [8, 9, 3, 2]]
            torso_heights = []
            if visible[13] and visible[2]:
                torso_heights.append(
                    np.linalg.norm(joints[:2, 13] - joints[:2, 2]))
            if visible[13] and visible[3]:
                torso_heights.append(
                    np.linalg.norm(joints[:2, 13] - joints[:2, 3]))
            # Make torso 75px
            if len(torso_heights) > 0:
                scale = 75. / np.mean(torso_heights)
            else:
                if visible[8] and visible[2]:
                    torso_heights.append(
                        np.linalg.norm(joints[:2, 8] - joints[:2, 2]))
                if visible[9] and visible[3]:
                    torso_heights.append(
                        np.linalg.norm(joints[:2, 9] - joints[:2, 3]))
                if len(torso_heights) > 0:
                    scale = 56. / np.mean(torso_heights)
                else:
                    # Skip, person is too close.
                    continue

        people.append((joints, scale, pos))

    return people


def add_to_tfrecord(anno, img_id, img_dir, coder, writer, is_train):
    """
    Add each "single person" in this image.
    anno - the entire annotation file.

    Returns:
      The number of people added.
    """
    anno_info = anno.annolist[img_id]
    # Make it consistent,, always a numpy array.
    single_persons = anno.single_person[img_id]
    if not isinstance(single_persons, np.ndarray):
        single_persons = np.array([single_persons])

    people = parse_people(anno_info, single_persons)

    if len(people) == 0:
        return 0

    # Add each people to tf record
    image_path = join(img_dir, anno_info.image.name)
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()
    image = coder.decode_jpeg(image_data)

    for joints, scale, pos in people:
        # Scale image:
        image_scaled, scale_factors = resize_img(image, scale)
        height, width = image_scaled.shape[:2]
        joints_scaled = np.copy(joints)
        joints_scaled[0, :] *= scale_factors[0]
        joints_scaled[1, :] *= scale_factors[1]

        visible = joints[2, :].astype(bool)
        min_pt = np.min(joints_scaled[:2, visible], axis=1)
        max_pt = np.max(joints_scaled[:2, visible], axis=1)
        center = (min_pt + max_pt) / 2.

        ## Crop 600x600 around this image..
        margin = 300
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

        # Encode image:
        image_data_scaled = coder.encode_jpeg(image_scaled)

        example = convert_to_example(image_data_scaled, image_path, height,
                                     width, joints_scaled, center)
        writer.write(example.SerializeToString())

    # Finally return how many were written.
    return len(people)


def process_mpii(anno, img_dir, out_dir, num_shards, is_train=True):
    all_ids = np.array(range(len(anno.annolist)))
    if is_train:
        out_path = join(out_dir, 'train_%03d.tfrecord')
        img_inds = all_ids[anno.img_train.astype('bool')]
    else:
        out_path = join(out_dir, 'test_%03d.tfrecord')
        img_inds = all_ids[np.logical_not(anno.img_train)]
        print('Not implemented for test data')
        exit(1)

    # MPII annotation is tricky (maybe the way scipy reads them)
    # If there's only 1 person in the image, annorect is not an array
    # So just go over each image, and add every single_person in that image
    # add_to_tfrecords returns the # of ppl added.
    # So it's possible some shards go over the limit but this is ok.

    coder = ImageCoder()

    i = 0
    # Count on shards
    fidx = 0
    num_ppl = 0
    while i < len(img_inds):

        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while i < len(img_inds) and num_ppl < num_shards:
                if i % 100 == 0:
                    print('Reading img %d/%d' % (i, len(img_inds)))
                num_ppl += add_to_tfrecord(anno, img_inds[i], img_dir, coder,
                                           writer, is_train)
                i += 1

        fidx += 1


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    anno_mat = join(FLAGS.img_directory, 'annotations',
                    'mpii_human_pose_v1_u12_1.mat')
    anno = load_anno(anno_mat)

    img_dir = join(FLAGS.img_directory, 'images')
    process_mpii(
        anno,
        img_dir,
        FLAGS.output_directory,
        FLAGS.train_shards,
        is_train=True)


if __name__ == '__main__':
    tf.app.run()
