"""
HMR trainer.
From an image input, trained a model that outputs 85D latent vector
consisting of [cam (3 - [scale, tx, ty]), pose (72), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import num_examples

from .ops import keypoint_l1_loss, compute_3d_loss, align_by_pelvis
from .models import Discriminator_separable_rotations, get_encoder_fn_separate

from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot

from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np

from os.path import join, dirname
import deepdish as dd

# For drawing
from .util import renderer as vis_util


class HMRTrainer(object):
    def __init__(self, config, data_loader, mocap_loader):
        """
        Args:
          config
          if no 3D label is available,
              data_loader is a dict
          else
              data_loader is a dict
        mocap_loader is a tuple (pose, shape)
        """
        # Config + path
        self.config = config
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        self.encoder_only = config.encoder_only
        self.use_3d_label = config.use_3d_label

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch

        self.num_cam = 3
        self.proj_fn = batch_orth_proj_idrot

        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Data
        num_images = num_examples(config.datasets)
        num_mocap = num_examples(config.mocap_datasets)

        self.num_itr_per_epoch = num_images / self.batch_size
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            data_loader['image'] = tf.transpose(data_loader['image'],
                                                [0, 3, 1, 2])

        self.image_loader = data_loader['image']
        self.kp_loader = data_loader['label']

        if self.use_3d_label:
            self.poseshape_loader = data_loader['label3d']
            # image_loader[3] is N x 2, first column is 3D_joints gt existence,
            # second column is 3D_smpl gt existence
            self.has_gt3d_joints = data_loader['has3d'][:, 0]
            self.has_gt3d_smpl = data_loader['has3d'][:, 1]

        self.pose_loader = mocap_loader[0]
        self.shape_loader = mocap_loader[1]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.log_img_step = config.log_img_step

        # For visualization:
        num2show = np.minimum(6, self.batch_size)
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.batch_size - np.arange(3) - 1]),
            tf.int32)

        # Model spec
        self.model_type = config.model_type
        self.keypoint_loss = keypoint_l1_loss

        # Optimizer, learning rate
        self.e_lr = config.e_lr
        self.d_lr = config.d_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd
        self.e_loss_weight = config.e_loss_weight
        self.d_loss_weight = config.d_loss_weight
        self.e_3d_weight = config.e_3d_weight

        self.optimizer = tf.train.AdamOptimizer

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)
        self.E_var = []
        self.build_model()

        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.pretrained_model_path)
            if 'resnet_v2_50' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v2_50' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            elif 'pose-tensorflow' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v1_101' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            else:
                self.pre_train_saver = tf.train.Saver()

            def load_pretrain(sess):
                self.pre_train_saver.restore(sess, self.pretrained_model_path)

            init_fn = load_pretrain

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            global_step=self.global_step,
            saver=self.saver,
            summary_writer=self.summary_writer,
            init_fn=init_fn)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        if ('resnet' in self.model_type) and (self.pretrained_model_path is
                                              not None):
            # Check is model_dir is empty
            import os
            if os.listdir(self.model_dir) == []:
                return True

        return False

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = join(
            dirname(self.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean

    def build_model(self):
        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        # Extract image features.
        self.img_feat, self.E_var = img_enc_fn(
            self.image_loader, weight_decay=self.e_wd, reuse=False)

        loss_kps = []
        if self.use_3d_label:
            loss_3d_joints, loss_3d_params = [], []
        # For discriminator
        fake_rotations, fake_shapes = [], []
        # Start loop
        # 85D
        theta_prev = self.load_mean_param()

        # For visualizations
        self.all_verts = []
        self.all_pred_kps = []
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_theta, _ = threed_enc_fn(
                    state, num_output=self.total_params, reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, Js, pred_Rs = self.smpl(shapes, poses, get_skin=True)
            pred_kp = batch_orth_proj_idrot(
                Js, cams, name='proj2d_stage%d' % i)
            # --- Compute losses:
            loss_kps.append(self.e_loss_weight * self.keypoint_loss(
                self.kp_loader, pred_kp))
            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])
            if self.use_3d_label:
                loss_poseshape, loss_joints = self.get_3d_loss(
                    pred_Rs, shapes, Js)
                loss_3d_params.append(loss_poseshape)
                loss_3d_joints.append(loss_joints)

            # Save pred_rotations for Discriminator
            fake_rotations.append(pred_Rs[:, 1:, :])
            fake_shapes.append(shapes)

            # Save things for visualiations:
            self.all_verts.append(tf.gather(verts, self.show_these))
            self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            self.all_pred_cams.append(tf.gather(cams, self.show_these))

            # Finally update to end iteration.
            theta_prev = theta_here

        if not self.encoder_only:
            self.setup_discriminator(fake_rotations, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_e_loss"):
            # Just the last loss.
            self.e_loss_kp = loss_kps[-1]

            if self.encoder_only:
                self.e_loss = self.e_loss_kp
            else:
                self.e_loss = self.d_loss_weight * self.e_loss_disc + self.e_loss_kp

            if self.use_3d_label:
                self.e_loss_3d = loss_3d_params[-1]
                self.e_loss_3d_joints = loss_3d_joints[-1]

                self.e_loss += (self.e_loss_3d + self.e_loss_3d_joints)

        if not self.encoder_only:
            with tf.name_scope("gather_d_loss"):
                self.d_loss = self.d_loss_weight * (
                    self.d_loss_real + self.d_loss_fake)

        # For visualizations, only save selected few into:
        # B x T x ...
        self.all_verts = tf.stack(self.all_verts, axis=1)
        self.all_pred_kps = tf.stack(self.all_pred_kps, axis=1)
        self.all_pred_cams = tf.stack(self.all_pred_cams, axis=1)
        self.show_imgs = tf.gather(self.image_loader, self.show_these)
        self.show_kps = tf.gather(self.kp_loader, self.show_these)

        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)

        # Setup optimizer
        print('Setting up optimizer..')
        d_optimizer = self.optimizer(self.d_lr)
        e_optimizer = self.optimizer(self.e_lr)

        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.global_step, var_list=self.E_var)
        if not self.encoder_only:
            self.d_opt = d_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.setup_summaries(loss_kps)

        print('Done initializing trainer!')

    def setup_summaries(self, loss_kps):
        # Prepare Summary
        always_report = [
            tf.summary.scalar("loss/e_loss_kp_noscale",
                              self.e_loss_kp / self.e_loss_weight),
            tf.summary.scalar("loss/e_loss", self.e_loss),
        ]
        if self.encoder_only:
            print('ENCODER ONLY!!!')
        else:
            always_report.extend([
                tf.summary.scalar("loss/d_loss", self.d_loss),
                tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
                tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
                tf.summary.scalar("loss/e_loss_disc",
                                  self.e_loss_disc / self.d_loss_weight),
            ])
        # loss at each stage.
        for i in np.arange(self.num_stage):
            name_here = "loss/e_losses_noscale/kp_loss_stage%d" % i
            always_report.append(
                tf.summary.scalar(name_here, loss_kps[i] / self.e_loss_weight))
        if self.use_3d_label:
            always_report.append(
                tf.summary.scalar("loss/e_loss_3d_params_noscale",
                                  self.e_loss_3d / self.e_3d_weight))
            always_report.append(
                tf.summary.scalar("loss/e_loss_3d_joints_noscale",
                                  self.e_loss_3d_joints / self.e_3d_weight))

        if not self.encoder_only:
            summary_occ = []
            # Report D output for each joint.
            smpl_names = [
                'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
                'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
                'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
                'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
                'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
                'Left_Finger', 'Right_Finger'
            ]
            # d_out is 25 (or 24), last bit is shape, first 24 is pose
            # 23(relpose) + 1(jointpose) + 1(shape) => 25
            d_out_pose = self.d_out[:, :24]
            for i, name in enumerate(smpl_names):
                summary_occ.append(
                    tf.summary.histogram("d_out/%s" % name, d_out_pose[i]))
            summary_occ.append(
                tf.summary.histogram("d_out/all_joints", d_out_pose[23]))
            summary_occ.append(
                tf.summary.histogram("d_out/beta", self.d_out[:, 24]))

            self.summary_op_occ = tf.summary.merge(
                summary_occ, collections=['occasional'])
        self.summary_op_always = tf.summary.merge(always_report)

    def setup_discriminator(self, fake_rotations, fake_shapes):
        # Compute the rotation matrices of "rea" pose.
        # These guys are in 24 x 3.
        real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
        real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
        # Ignoring global rotation. N x 23*9
        # The # of real rotation is B*num_stage so it's balanced.
        real_rotations = real_rotations[:, 1:, :]
        all_fake_rotations = tf.reshape(
            tf.concat(fake_rotations, 0),
            [self.batch_size * self.num_stage, -1, 9])
        comb_rotations = tf.concat(
            [real_rotations, all_fake_rotations], 0, name="combined_pose")

        comb_rotations = tf.expand_dims(comb_rotations, 2)
        all_fake_shapes = tf.concat(fake_shapes, 0)
        comb_shapes = tf.concat(
            [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.d_wd,
            'shapes': comb_shapes,
            'poses': comb_rotations
        }

        self.d_out, self.D_var = Discriminator_separable_rotations(
            **disc_input)

        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)
        # Compute losses:
        with tf.name_scope("comp_d_loss"):
            self.d_loss_real = tf.reduce_mean(
                tf.reduce_sum((self.d_out_real - 1)**2, axis=1))
            self.d_loss_fake = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake)**2, axis=1))
            # Encoder loss
            self.e_loss_disc = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake - 1)**2, axis=1))

    def get_3d_loss(self, Rs, shape, Js):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        Rs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([Rs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshape_loader[:, :226]
        loss_poseshape = self.e_3d_weight * compute_3d_loss(
            params_pred, gt_params, self.has_gt3d_smpl)
        # 14*3 = 42
        gt_joints = self.poseshape_loader[:, 226:]
        pred_joints = Js[:, :14, :]
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])
        gt_joints = tf.reshape(gt_joints, [self.batch_size, 14, 3])
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])

        loss_joints = self.e_3d_weight * compute_3d_loss(
            pred_joints, gt_joints, self.has_gt3d_joints)

        return loss_poseshape, loss_joints

    def visualize_img(self, img, gt_kp, vert, pred_kp, cam, renderer):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """
        gt_vis = gt_kp[:, 2].astype(bool)
        loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
        debug_text = {"sc": cam[0], "tx": cam[1], "ty": cam[2], "kpl": loss}
        # Fix a flength so i can render this with persp correct scale
        f = 5.
        tz = f / cam[0]
        cam_for_render = 0.5 * self.img_size * np.array([f, 1, 1])
        cam_t = np.array([cam[1], cam[2], tz])
        # Undo pre-processing.
        input_img = (img + 1) * 0.5
        rend_img = renderer(vert + cam_t, cam_for_render, img=input_img)
        rend_img = vis_util.draw_text(rend_img, debug_text)

        # Draw skeleton
        gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * self.img_size
        pred_joint = ((pred_kp + 1) * 0.5) * self.img_size
        img_with_gt = vis_util.draw_skeleton(
            input_img, gt_joint, draw_edges=False, vis=gt_vis)
        skel_img = vis_util.draw_skeleton(img_with_gt, pred_joint)

        combined = np.hstack([skel_img, rend_img / 255.])

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(skel_img)
        # import ipdb; ipdb.set_trace()
        return combined

    def draw_results(self, result):
        from StringIO import StringIO
        import matplotlib.pyplot as plt

        # This is B x H x W x 3
        imgs = result["input_img"]
        # B x 19 x 3
        gt_kps = result["gt_kp"]
        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])
        # This is B x T x 6890 x 3
        est_verts = result["e_verts"]
        # B x T x 19 x 2
        joints = result["joints"]
        # B x T x 3
        cams = result["cam"]

        img_summaries = []

        for img_id, (img, gt_kp, verts, joints, cams) in enumerate(
                zip(imgs, gt_kps, est_verts, joints, cams)):
            # verts, joints, cams are a list of len T.
            all_rend_imgs = []
            for vert, joint, cam in zip(verts, joints, cams):
                rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                              self.renderer)
                all_rend_imgs.append(rend_img)
            combined = np.vstack(all_rend_imgs)

            sio = StringIO()
            plt.imsave(sio, combined, format='png')
            vis_sum = tf.Summary.Image(
                encoded_image_string=sio.getvalue(),
                height=combined.shape[0],
                width=combined.shape[1])
            img_summaries.append(
                tf.Summary.Value(tag="vis_images/%d" % img_id, image=vis_sum))

        img_summary = tf.Summary(value=img_summaries)
        self.summary_writer.add_summary(
            img_summary, global_step=result['step'])

    def train(self):
        # For rendering!
        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=self.config.smpl_face_path)

        step = 0

        with self.sv.managed_session(config=self.sess_config) as sess:
            while not self.sv.should_stop():
                fetch_dict = {
                    "summary": self.summary_op_always,
                    "step": self.global_step,
                    "e_loss": self.e_loss,
                    # The meat
                    "e_opt": self.e_opt,
                    "loss_kp": self.e_loss_kp
                }
                if not self.encoder_only:
                    fetch_dict.update({
                        # For D:
                        "d_opt": self.d_opt,
                        "d_loss": self.d_loss,
                        "loss_disc": self.e_loss_disc,
                    })
                if self.use_3d_label:
                    fetch_dict.update({
                        "loss_3d_params": self.e_loss_3d,
                        "loss_3d_joints": self.e_loss_3d_joints
                    })

                if step % self.log_img_step == 0:
                    fetch_dict.update({
                        "input_img": self.show_imgs,
                        "gt_kp": self.show_kps,
                        "e_verts": self.all_verts,
                        "joints": self.all_pred_kps,
                        "cam": self.all_pred_cams,
                    })
                    if not self.encoder_only:
                        fetch_dict.update({
                            "summary_occasional":
                            self.summary_op_occ
                        })

                t0 = time()
                result = sess.run(fetch_dict)
                t1 = time()

                self.summary_writer.add_summary(
                    result['summary'], global_step=result['step'])

                e_loss = result['e_loss']
                step = result['step']

                epoch = float(step) / self.num_itr_per_epoch
                if self.encoder_only:
                    print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                          (step, epoch, t1 - t0, e_loss))
                else:
                    d_loss = result['d_loss']
                    print(
                        "itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f"
                        % (step, epoch, t1 - t0, e_loss, d_loss))

                if step % self.log_img_step == 0:
                    if not self.encoder_only:
                        self.summary_writer.add_summary(
                            result['summary_occasional'],
                            global_step=result['step'])
                    self.draw_results(result)

                self.summary_writer.flush()
                if epoch > self.max_epoch:
                    self.sv.request_stop()

                step += 1

        print('Finish training on %s' % self.model_dir)
