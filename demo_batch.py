"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import os
import pickle
from tqdm import tqdm

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string(
    'result_path', 'results',
    'Folder to save estimated parameters.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()

    
def save_obj(verts, obj_path):
    #cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
    #    proc_param, verts, cam, joints, img_size=img.shape[:2])
        
    verts[:,1] *= -1
    smpl_faces = np.load('src/tf_smpl/smpl_faces.npy')
    #print(smpl_faces.shape, smpl_faces.min())
    
    file = open(obj_path, 'w')
    for v in verts:
        file.write('v %10.6f %10.6f %10.6f\n' % (v[0], v[1], v[2]))
        
    for f in smpl_faces:
        file.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))
        
    file.close()

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, result_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
        
    if os.path.isdir(img_path):
        imgs = os.listdir(img_path)
        jsons = [None] * len(imgs)
        for i, im in enumerate(imgs):
            json = os.path.join(json_path, im.replace('.jpg', '_keypoints.json'))
            if os.path.isfile(json):
                jsons[i] = json
            
        imgs = [os.path.join(img_path, item) for item in imgs]
    else:
        imgs = [img_path]
        jsons = [None] if json_path is None else [json_path]
            
    for _IMG, _JSON in tqdm(zip(imgs, jsons)):
        input_img, proc_param, img = preprocess_image(_IMG, _JSON)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)
        
        pic = {
            'joints': joints[0],
            'verts': verts[0],
            'cams': cams[0],
            'joints3d': joints3d[0],
            'theta_cam': theta[0][:3],
            'pose': theta[0][3:-10],
            'shape': theta[0][-10:]
        }
        name = _IMG.split('/')[-1].split('.')[0] + '.pickle'
        with open(os.path.join(result_path, name), 'wb') as file:
            pickle.dump(pic, file)
        

    #visualize(img, proc_param, joints[0], verts[0], cams[0])
    #save_obj(verts[0], 'out.obj')
    
if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.result_path, config.json_path)
