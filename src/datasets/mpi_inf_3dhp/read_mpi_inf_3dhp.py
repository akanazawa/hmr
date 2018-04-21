"""
Open up mpi_inf_3dhp.

TRAINING:
For each subject & sequence there is annot.mat
What is in annot.mat:
  'frames': number of frames, N
  'univ_annot3': (14,) for each camera of N x 84 -> Why is there univ for each camera if it's univ..?
  'annot3': (14,) for each camera of N x 84
  'annot2': (14,) for each camera of N x 56
  'cameras':

  In total there are 28 joints, but H3.6M subsets are used.

  The image frames are unpacked in:
  BASE_DIR/S%d/Seq%d/video_%d/frame_%06.jpg


TESTING:
  'valid_frame': N_frames x 1
  'annot2': N_frames x 1 x 17 x 2
  'annot3': N_frames x 1 x 17 x 3
  'univ_annot3': N_frames x 1 x 17 x 3
  'bb_crop': this is N_frames x 34 (not sure what this is..)
  'activity_annotation': N_frames x 1 (of integer indicating activity type
  The test images are already in jpg.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from os.path import join


def get_paths(base_dir, sub_id, seq_id):
    data_dir = join(base_dir, 'S%d' % sub_id, 'Seq%d' % seq_id)
    anno_path = join(data_dir, 'annot.mat')
    img_dir = join(data_dir, 'imageFrames')
    return img_dir, anno_path


def read_mat(path):
    from scipy.io import loadmat
    res = loadmat(path, struct_as_record=True, squeeze_me=True)

    cameras = res['cameras']
    annot2 = np.stack(res['annot2'])
    annot3 = np.stack(res['annot3'])
    frames = res['frames']

    # univ_annot3 = np.stack(res['univ_annot3'])

    return frames, cameras, annot2, annot3


def mpi_inf_3dhp_to_lsp_idx():
    # For training, this joint_idx gives names 17
    raw_to_h36m17_idx = np.array(
        [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]) - 1
    names_17 = [
        'Head', 'Neck', 'R Shoulder', 'R Elbow', 'R Wrist', 'L Shoulder',
        'L Elbow', 'L Wrist', 'R Hip', 'R Knee', 'R Ankle', 'L Hip', 'L Knee',
        'L Ankle', 'Pelvis', 'Spine', 'Head'
    ]
    want_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head'
    ]

    h36m17_to_lsp_idx = [names_17.index(j) for j in want_names]

    raw_to_lsp_idx = raw_to_h36m17_idx[h36m17_to_lsp_idx]

    return raw_to_lsp_idx, h36m17_to_lsp_idx


def read_camera(base_dir):
    cam_path = join(base_dir, 'S1/Seq1/camera.calibration')
    lines = []
    with open(cam_path, 'r') as f:
        for line in f:
            content = [x for x in line.strip().split(' ') if x]
            lines.append(content)

    def get_cam_info(block):
        cam_id = int(block[0][1])
        # Intrinsic
        intrinsic = block[4][1:]
        K = np.array([np.float(cont) for cont in intrinsic]).reshape(4, 4)
        # Extrinsic:
        extrinsic = block[5][1:]
        Ext = np.array([float(cont) for cont in extrinsic]).reshape(4, 4)
        return cam_id, K, Ext

    # Skip header
    lines = lines[1:]
    # each camera is 7 lines long.
    num_cams = int(len(lines) / 7)
    cams = {}
    for i in range(num_cams):
        cam_id, K, Ext = get_cam_info(lines[7 * i:7 * i + 7])
        cams[cam_id] = K

    return cams
