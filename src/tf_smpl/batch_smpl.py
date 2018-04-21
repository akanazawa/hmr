""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle

import tensorflow as tf
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)    
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = tf.Variable(
            dd['J_regressor'].T.todense(),
            name="J_regressor",
            dtype=dtype,
            trainable=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=dtype,
            trainable=False)

        # This returns 19 keypoints: 6890 x 19
        self.joint_regressor = tf.Variable(
            dd['cocoplus_regressor'].T.todense(),
            name="cocoplus_regressor",
            dtype=dtype,
            trainable=False)
        if joint_type == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()

    def __call__(self, beta, theta, get_skin=False, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta]):
            num_batch = beta.shape[0].value

            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            v_shaped = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template

            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3),
                                          [-1, 207])

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + v_shaped

            #4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

            # 5. Do skinning:
            # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]

            # Get cocoplus or lsp joints:
            joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
            joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
            joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
            joints = tf.stack([joint_x, joint_y, joint_z], axis=2)

            if get_skin:
                return verts, joints, Rs
            else:
                return joints


