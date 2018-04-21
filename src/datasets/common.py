"""
Helpers for tfrecord conversion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(image_data, image_path, height, width, label, center):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      image_path: string, path to this image file
      labels: 3 x 14 joint location + visibility --> This could be 3 x 19
      height, width: integers, image shapes in pixels.
      center: 2 x 1 center of the tight bbox
    Returns:
      Example proto
    """
    from os.path import basename

    image_format = 'JPEG'
    add_face = False
    if label.shape[1] == 19:
        add_face = True
        # Split and save facepts on it's own.
        face_pts = label[:, 14:]
        label = label[:, :14]

    feat_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/center': int64_feature(center.astype(np.int)),
        'image/x': float_feature(label[0, :].astype(np.float)),
        'image/y': float_feature(label[1, :].astype(np.float)),
        'image/visibility': int64_feature(label[2, :].astype(np.int)),
        'image/format': bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': bytes_feature(
            tf.compat.as_bytes(basename(image_path))),
        'image/encoded': bytes_feature(tf.compat.as_bytes(image_data)),
    }
    if add_face:
        # 3 x 5
        feat_dict.update({
            'image/face_pts':
            float_feature(face_pts.ravel().astype(np.float))
        })

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example


def convert_to_example_wmosh(image_data, image_path, height, width, label,
                             center, gt3d, pose, shape, scale_factors,
                             start_pt, cam):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      image_path: string, path to this image file
      labels: 3 x 14 joint location + visibility
      height, width: integers, image shapes in pixels.
      center: 2 x 1 center of the tight bbox
      gt3d: 14x3 3D joint locations
      scale_factors: 2 x 1, scale factor used to scale image.
      start_pt: the left corner used to crop the _scaled_ image to 300x300
      cam: (3,), [f, px, py] intrinsic camera parameters.
    Returns:
      Example proto
    """
    from os.path import basename
    image_format = 'JPEG'
    if label.shape[0] != 3:
        label = label.T
    if label.shape[1] > 14:
        print('This shouldnt be happening')
        import ipdb
        ipdb.set_trace()
    if pose is None:
        has_3d = 0
        # Use -1 to save.
        pose = -np.ones(72)
        shape = -np.ones(10)
    else:
        has_3d = 1

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':
            int64_feature(height),
            'image/width':
            int64_feature(width),
            'image/center':
            int64_feature(center.astype(np.int)),
            'image/x':
            float_feature(label[0, :].astype(np.float)),
            'image/y':
            float_feature(label[1, :].astype(np.float)),
            'image/visibility':
            int64_feature(label[2, :].astype(np.int)),
            'image/format':
            bytes_feature(tf.compat.as_bytes(image_format)),
            'image/filename':
            bytes_feature(tf.compat.as_bytes(basename(image_path))),
            'image/encoded':
            bytes_feature(tf.compat.as_bytes(image_data)),
            'mosh/pose':
            float_feature(pose.astype(np.float)),
            'mosh/shape':
            float_feature(shape.astype(np.float)),
            'mosh/gt3d':
            float_feature(gt3d.ravel().astype(np.float)),
            'meta/scale_factors':
            float_feature(np.array(scale_factors).astype(np.float)),
            'meta/crop_pt':
            int64_feature(start_pt.astype(np.int)),
            'meta/has_3d':
            int64_feature(has_3d),
            'image/cam':
            float_feature(cam.astype(np.float)),
        }))

    return example


def resize_img(img, scale_factor):
    import cv2
    import numpy as np
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor
