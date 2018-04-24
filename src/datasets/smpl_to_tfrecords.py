"""
Convert MoCap SMPL data to tfrecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs
from os.path import join, exists
import numpy as np
from glob import glob
import cPickle as pickle

import tensorflow as tf

from .common import float_feature

tf.app.flags.DEFINE_string(
    'dataset_name', 'neutrSMPL_CMU',
    'neutrSMPL_CMU, neutrSMPL_H3.6, or neutrSMPL_jointLim')
tf.app.flags.DEFINE_string('data_directory',
                           '/scratch1/storage/human_datasets/neutrMosh/',
                           'data directory where SMPL npz/pkl lies')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch1/projects/tf_datasets/mocap_neutrMosh/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_shards', 10000,
                            'Number of shards in TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def convert_to_example(pose, shape=None):
    """Build an Example proto for an image example.
    Args:
      pose: 72-D vector, float
     shape: 10-D vector, float
    Returns:
      Example proto
    """
    if shape is None:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pose': float_feature(pose.astype(np.float))
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pose': float_feature(pose.astype(np.float)),
                'shape': float_feature(shape.astype(np.float)),
            }))

    return example


def process_smpl_mocap(all_pkls, out_dir, num_shards, dataset_name):
    all_poses, all_shapes, all_shapes_unique = [], [], []
    for pkl in all_pkls:
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
            if 'poses' in res.keys():
                all_poses.append(res['poses'])
                num_poses_here = res['poses'].shape[0]
            else:
                all_poses.append(res['new_poses'])
                num_poses_here = res['new_poses'].shape[0]
            all_shapes.append(
                np.tile(np.reshape(res['betas'], (10, 1)), num_poses_here))
            all_shapes_unique.append(res['betas'])

    all_poses = np.vstack(all_poses)
    all_shapes = np.hstack(all_shapes).T

    out_path = join(out_dir, '%s_%%03d.tfrecord' % dataset_name)

    # shuffle results
    num_mocap = all_poses.shape[0]
    shuffle_id = np.random.permutation(num_mocap)
    all_poses = all_poses[shuffle_id]
    all_shapes = all_shapes[shuffle_id]

    i = 0
    fidx = 0
    while i < num_mocap:
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < num_mocap and j < num_shards:
                if i % 10000 == 0:
                    print('Converting mosh %d/%d' % (i, num_mocap))
                example = convert_to_example(all_poses[i], shape=all_shapes[i])
                writer.write(example.SerializeToString())
                i += 1
                j += 1

        fidx += 1


def main(unused_argv):
    data_dir = join(FLAGS.data_directory, FLAGS.dataset_name)
    # Ignore H3.6M test subjects!!
    all_pkl = sorted([
        f for f in glob(join(data_dir, '*/*.pkl'))
        if 'S9' not in f and 'S11' not in f
    ])
    if len(all_pkl) == 0:
        print('Something is wrong with the path bc I cant find any pkls!')
        import ipdb; ipdb.set_trace()

    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    process_smpl_mocap(all_pkl, FLAGS.output_directory, FLAGS.num_shards,
                       FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()
