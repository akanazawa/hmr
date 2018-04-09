""" Util functions for SMPL
@@batch_skew
@@batch_rodrigues
@@batch_lrotmin
@@batch_global_rigid_transformation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    with tf.name_scope("batch_skew", [vec]):
        if batch_size is None:
            batch_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res


def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """
    with tf.name_scope(name, "batch_rodrigues", [theta]):
        batch_size = theta.shape.as_list()[0]

        # angle = tf.norm(theta, axis=1)
        # r = tf.expand_dims(tf.div(theta, tf.expand_dims(angle + 1e-8, -1)), -1)
        # angle = tf.expand_dims(tf.norm(theta, axis=1) + 1e-8, -1)
        angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = tf.expand_dims(tf.div(theta, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
        return R


def batch_lrotmin(theta, name=None):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
    with tf.name_scope(name, "batch_lrotmin", [theta]):
        with tf.name_scope("ignore_global"):
            theta = theta[:, 3:]

        # N*23 x 3 x 3
        Rs = batch_rodrigues(tf.reshape(theta, [-1, 3]))
        lrotmin = tf.reshape(Rs - tf.eye(3), [-1, 207])

        return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    with tf.name_scope("batch_forward_kinematics", [Rs, Js]):
        N = Rs.shape[0].value
        if rotate_base:
            print('Flipping the SMPL coordinate frame!!!!')
            rot_x = tf.constant(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
            rot_x = tf.reshape(tf.tile(rot_x, [N, 1]), [N, 3, 3])
            root_rotation = tf.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]

        # Now Js is N x 24 x 3 x 1
        Js = tf.expand_dims(Js, -1)

        def make_A(R, t, name=None):
            # Rs is N x 3 x 3, ts is N x 3 x 1
            with tf.name_scope(name, "Make_A", [R, t]):
                R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])
                t_homo = tf.concat([t, tf.ones([N, 1, 1])], 1)
                return tf.concat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = tf.matmul(
                results[parent[i]], A_here, name="propA%d" % i)
            results.append(res_here)

        # 10 x 24 x 4 x 4
        results = tf.stack(results, axis=1)

        new_J = results[:, :, :3, 3]

        # --- Compute relative A: Skinning is based on
        # how much the bone moved (not the final location of the bone)
        # but (final_bone - init_bone)
        # ---
        Js_w0 = tf.concat([Js, tf.zeros([N, 24, 1, 1])], 2)
        init_bone = tf.matmul(results, Js_w0)
        # Append empty 4 x 3:
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone

        return new_J, A
