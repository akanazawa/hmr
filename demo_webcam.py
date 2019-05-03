import numpy as np
import cv2

import vispy
import vispy.scene
from vispy.scene import visuals


import tensorflow as tf
import src.config
import sys
from absl import flags


from src.util import image as img_util
from src.RunModel import RunModel
import datetime


def preprocess_image(img):

    if np.max(img.shape[:2]) != config.img_size:
        # print('Resizing so the max image size is %d..' % img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop

def main():
    # Video capture
    cap = cv2.VideoCapture(0)

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # create scatter object
    scatter = visuals.Markers()

    # generate data or figure out how to prevent crash without data ^^
    pos = np.random.normal(size=(100000, 3), scale=0.2)
    scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)

    view.add(scatter)

    #configure view
    view.camera = 'turntable'  # or try 'arcball'
    axis = visuals.XYZAxis(parent=view.scene)

    #load model
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()


        processed = preprocess_image(frame)

        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(processed, 0)
        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        start = datetime.datetime.now()
        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)
        end = datetime.datetime.now()
        delta = end -start
        print("took:" , delta)

        # Display Camera frame
        cv2.imshow('frame',frame)
        cv2.imshow('processed',processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Display Plot
        # pos = np.random.normal(size=(100000, 3), scale=0.2)
        scatter.set_data(verts[0], edge_color=None, face_color=(1, 1, 1, .5), size=5)


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    # renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main()