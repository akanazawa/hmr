"""
Convert LSP/LSP extended to TFRecords.
In LSP, the first 1000 is training and the last 1000 is test/validation.
All of LSP extended is training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs
from os.path import join, exists
from glob import glob

import numpy as np

import tensorflow as tf

from .common import convert_to_example, ImageCoder

tf.app.flags.DEFINE_string('img_directory',
                           '/scratch1/storage/human_datasets/lsp_dataset',
                           'image data directory')
tf.app.flags.DEFINE_string(
    'output_directory', '/Users/kanazawa/projects/datasets/tf_datasets/lsp/',
    'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 500,
                            'Number of shards in validation TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def _add_to_tfrecord(image_path, label, coder, writer, is_lsp_ext=False):
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()

    image = coder.decode_jpeg(image_data)
    height, width = image.shape[:2]
    assert image.shape[2] == 3

    # LSP 3-D dim, 0 means visible 1 means invisible.
    # But in LSP-ext, 0 means invis, 1 means visible
    # Negate this
    if is_lsp_ext:
        visible = label[2, :].astype(bool)
    else:
        visible = np.logical_not(label[2, :])
        label[2, :] = visible.astype(label.dtype)
    min_pt = np.min(label[:2, visible], axis=1)
    max_pt = np.max(label[:2, visible], axis=1)
    center = (min_pt + max_pt) / 2.
    """
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.imshow(image)
    plt.scatter(label[0, visible], label[1, visible])
    plt.scatter(center[0], center[1])
    # bwidth, bheight = max_pt - min_pt + 1
    # rect = plt.Rectangle(min_pt, bwidth, bheight, fc='None', ec='green')
    # ax.add_patch(rect)
    import ipdb; ipdb.set_trace()
    """

    example = convert_to_example(image_data, image_path, height, width, label,
                                 center)

    writer.write(example.SerializeToString())


def package(img_paths, labels, out_path, num_shards):
    """
    packages the images and labels into multiple tfrecords.
    """
    is_lsp_ext = True if len(img_paths) == 10000 else False
    coder = ImageCoder()

    i = 0
    fidx = 0
    while i < len(img_paths):
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(img_paths) and j < num_shards:
                if i % 100 == 0:
                    print('Converting image %d/%d' % (i, len(img_paths)))
                _add_to_tfrecord(
                    img_paths[i],
                    labels[:, :, i],
                    coder,
                    writer,
                    is_lsp_ext=is_lsp_ext)
                i += 1
                j += 1

        fidx += 1


def load_mat(fname):
    import scipy.io as sio
    res = sio.loadmat(fname)
    # this is 3 x 14 x 2000
    return res['joints']


def process_lsp(img_dir, out_dir, num_shards_train, num_shards_test):
    """Process a complete data set and save it as a TFRecord.
    LSP has 2000 images, first 1000 is train, last 1000 is test.

    Args:
      img_dir: string, root path to the data set.
      num_shards: integer number of shards for this data set.
    """
    # Load labels 3 x 14 x N
    labels = load_mat(join(img_dir, 'joints.mat'))
    if labels.shape[0] != 3:
        labels = np.transpose(labels, (1, 0, 2))

    all_images = sorted([f for f in glob(join(img_dir, 'images/*.jpg'))])

    if len(all_images) == 10000:
        # LSP-extended is all train.
        train_out = join(out_dir, 'train_%03d.tfrecord')
        package(all_images, labels, train_out, num_shards_train)
    else:
        train_out = join(out_dir, 'train_%03d.tfrecord')

        package(all_images[:1000], labels[:, :, :1000], train_out,
                num_shards_train)

        test_out = join(out_dir, 'test_%03d.tfrecord')
        package(all_images[1000:], labels[:, :, 1000:], test_out,
                num_shards_test)


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)
    process_lsp(FLAGS.img_directory, FLAGS.output_directory,
                FLAGS.train_shards, FLAGS.validation_shards)


if __name__ == '__main__':
    tf.app.run()
