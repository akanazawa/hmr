"""
Defines networks.

@Encoder_resnet
@Encoder_resnet_v1_101
@Encoder_fc3_dropout

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer

def Encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.name_scope("Encoder_resnet", [x]):
        with slim.arg_scope(
                resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            net, end_points = resnet_v2.resnet_v2_50(
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v2_50')
            net = tf.squeeze(net, axis=[1, 2])
    variables = tf.contrib.framework.get_variables('resnet_v2_50')
    return net, variables

def Encoder_resnet_v1_101(x,
                          weight_decay,
                          is_training=True,
                          reuse=False):
    """
    Resnet v1-101 encoder, adds 2 fc layers after Resnet.
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool-> True if test

    Outputs:
    - net: N x F
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v1
    with tf.name_scope("Encoder_resnet_v1_101", [x]):
        with slim.arg_scope(
                resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            net, end_points = resnet_v1.resnet_v1_101(
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v1_101')
            net = tf.reshape(net, [net.shape.as_list()[0], -1])
    variables = tf.contrib.framework.get_variables('resnet_v1_101')
    return net, variables

def Encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables


def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet_v1_101' in model_type:
        encoder_fn = Encoder_resnet_v1_101
    elif 'resnet' in model_type:
        encoder_fn = Encoder_resnet
        
    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout
        
    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        import ipdb; ipdb.set_trace()

    return encoder_fn, threed_fn
