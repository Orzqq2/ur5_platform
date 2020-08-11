""" This file defines Meta Imitation Learning (MIL). """
from __future__ import division

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from tf_utils import *

FLAGS = flags.FLAGS


class MIL(object):
    """ Initialize MIL. Need to call init_network to contruct the architecture after init. """

    def __init__(self, num_update=3):

        # MIL hyperparams
        self.activation_fn = tf.nn.relu
        self.num_updates = num_update
        self.clip_min = 10
        self.clip_max = -10
        self.step_size = 1e-3
        self.meta_batch_size = 3
        self.meta_lr = 0.001
        self.act_loss_eps = 0.9
        self.prob = 0.6
        self.arccos_size = 5

    def init_network(self, graph, prefix='Training_', meta_learning='Meta_'):
        """ Helper method to initialize the tf networks used """
        if 'Meta' in meta_learning:
            with graph.as_default():
                result = self.construct_model()
                outputa, outputbs, test_output_action, lossa, losseb = result
                if 'Testing' in prefix:
                    self.test_act_op = test_output_action

                total_losses2 = [tf.reduce_sum(losseb[j]) / tf.to_float(self.meta_batch_size) for j in
                                 range(self.num_updates)]

                if 'Training' in prefix:
                    self.total_losses2 = total_losses2
                    self.test_output_action = test_output_action
                elif 'Validation' in prefix:
                    self.val_total_losses2 = total_losses2

                if 'Training' in prefix:
                    self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(
                        self.total_losses2[self.num_updates - 1])
        else:
            with graph.as_default():
                result = self.construct_nn_model()
                lossa, arm_action, grip_action = result
                self.lossa = lossa
                self.arm_action = arm_action
                self.grip_action = grip_action
                self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.lossa)

    def construct_weights(self, dim_input=27, dim_output=7):
        """ Construct weights for the network. """
        weights = {}
        # Define the number of base layers, filter size, and step length
        # Image Conv Processing
        n_conv_layers = self.n_conv_layers = 3  # 5-layer convolution
        filter_sizes = [1, 3, 3]
        strides = [1, 2, 2, 1]
        im_height = 100
        im_width = 130
        im_num_channels = 4
        fan_in = im_num_channels - 1
        num_filters = [32, 32, 16]

        # Depth images conv process
        weights['wc_dep'] = init_conv_weights_snn([7, 7, 1, 16], name='wc_dep')
        weights['bc_dep'] = init_bias([16], name='bc_dep')

        # RGB images conv process
        weights['wc_rgb'] = init_conv_weights_snn([7, 7, fan_in, 64], name='wc_rgb')
        weights['bc_rgb'] = init_bias([64], name='bc_rgb')

        # After concat conv is [batch, 65, 50, 80]
        fan_in = 16 + 64
        for i in range(n_conv_layers):
            weights['wc%d' % (i + 1)] = init_conv_weights_snn(
                [filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]],
                name='wc%d' % (i + 1))
            weights['bc%d' % (i + 1)] = init_bias([num_filters[i]], name='bc%d' % (i + 1))
            fan_in = num_filters[i]
        self.shape_of_out_conv = [im_height // (2 ** 3) + 1, im_width // (2 ** 3) + 1, num_filters[-1]]

        # Image FC Processing
        fc_in_shape = self.fc_in_shape = int(reduce(lambda x, y: x * y, self.shape_of_out_conv))
        print fc_in_shape
        self.num_conv_fc_layer = 2
        num_fc_filters = [512, 64]
        for i in range(self.num_conv_fc_layer):
            weights['w_1d_conv%d' % i] = init_fc_weights_snn([fc_in_shape, num_fc_filters[i]], name='w_1d_conv%d' % i)
            weights['b_1d_conv%d' % i] = init_bias(num_fc_filters[i], name='b_1d_conv%d' % i)
            fc_in_shape = num_fc_filters[i]

        # auxiliary process
        weights['w_aux_fc'] = init_fc_weights_snn([64, 3], name='w_aux_fc')
        weights['b_aux_fc'] = init_bias(3, name='w_aux_fc')

        # joint FC process
        joint_fc_in = 7
        self.num_joint_fc_layer = num_joint_fc_layer = 1
        num_joint_filters = [64]
        for i in range(num_joint_fc_layer):
            weights['w_joint_fc%d' % i] = init_fc_weights_snn([joint_fc_in, num_joint_filters[i]],
                                                              name='w_joint_fc%d' % i)
            weights['b_joint_fc%d' % i] = init_bias(num_joint_filters[i], name='b_joint_fc%d' % i)

        # Image and Joint FC Processing
        im_joint_fc_in = 64 + 64
        self.num_im_joint_fc_layer = num_im_joint_fc_layer = 2
        num_im_joint_filters = [64, 6]  # delete grip action
        for i in range(num_im_joint_fc_layer):
            weights['w_im_joint_fc%d' % i] = init_fc_weights_snn([im_joint_fc_in, num_im_joint_filters[i]],
                                                                 name='w_im_joint_fc%d' % i)
            weights['b_im_joint_fc%d' % i] = init_bias(num_im_joint_filters[i], name='b_im_joint_fc%d' % i)
            im_joint_fc_in = num_im_joint_filters[i]
        weights['w_im_grip_fc'] = init_fc_weights_snn([64, 1], name='w_im_grip_fc')
        weights['b_im_grip_fc'] = init_bias(1, name='b_im_grip_fc')

        return weights

    # def construct_fc_weights(self, dim_input=27, dim_output=7, network_config=None):

    def forward(self, image_input, state_input, weights, meta_testing=False, is_training=True, testing=False,
                network_config=None):
        """ Perform the forward pass. """
        strides = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1]]
        stride = [1, 2, 2, 1]
        # RGB Conv process
        conv_RGB_layer = image_input[:, :, :, :3]
        conv_RGB_layer = dropout(
            norm(conv2d(img=conv_RGB_layer, w=weights['wc_rgb'], b=weights['bc_rgb'], strides=stride,
                        is_dilated=False), id=5, decay=0.9, is_training=is_training, activation_fn=tf.nn.relu),
            keep_prob=self.prob, is_training=is_training, name='dropout_conv_rgb')

        # Depth Conv process
        conv_depth_layer = image_input[:, :, :, 3]
        conv_depth_layer = tf.expand_dims(conv_depth_layer, -1)
        conv_depth_layer = dropout(
            norm(conv2d(img=conv_depth_layer, w=weights['wc_dep'], b=weights['bc_dep'], strides=stride,
                        is_dilated=False), id=6, decay=0.9, is_training=is_training, activation_fn=tf.nn.relu),
            keep_prob=self.prob, is_training=is_training, name='dropout_conv_depth')

        # Concat
        conv_layer = tf.concat([conv_RGB_layer, conv_depth_layer], -1)

        for i in range(self.n_conv_layers):
            conv_layer = dropout(norm(
                conv2d(img=conv_layer, w=weights['wc%d' % (i + 1)], b=weights['bc%d' % (i + 1)], strides=strides[i],
                       is_dilated=False), decay=0.9, id=i, is_training=is_training, activation_fn=tf.nn.relu),
                keep_prob=self.prob, is_training=is_training, name='dropout_conv_%d' % (i + 1))

        # Conv fc process, output shape is [-1, 64]
        conv_fc_ouput = tf.reshape(conv_layer, [-1, self.fc_in_shape])
        for i in range(self.num_conv_fc_layer):
            conv_fc_ouput = tf.matmul(conv_fc_ouput, weights['w_1d_conv%d' % i]) + weights['b_1d_conv%d' % i]
            conv_fc_ouput = self.activation_fn(conv_fc_ouput)
            conv_fc_ouput = dropout(conv_fc_ouput, keep_prob=self.prob, is_training=is_training,
                                    name='dropout_conv_fc_%d' % i)

        # Auxiliary process
        fc_aux_layer = tf.matmul(conv_fc_ouput, weights['w_aux_fc']) + weights['b_aux_fc']

        # Joint fc process, output shape is [-1, 64]
        joint_layer = state_input  # [-1, 7]
        for i in range(self.num_joint_fc_layer):
            joint_layer = tf.matmul(joint_layer, weights['w_joint_fc%d' % i]) + weights['b_joint_fc%d' % i]
            joint_layer = self.activation_fn(joint_layer)
            joint_layer = dropout(joint_layer, keep_prob=self.prob, is_training=is_training,
                                  name='dropout_joint_fc_%d' % i)

        fc_layer = tf.concat([conv_fc_ouput, joint_layer], 1)  # [-1, 128]
        # FC process, output is [-1, 6] (arm action)
        for i in range(self.num_im_joint_fc_layer - 1):
            fc_layer = tf.matmul(fc_layer, weights['w_im_joint_fc%d' % i]) + weights['b_im_joint_fc%d' % i]
            fc_layer = self.activation_fn(fc_layer)
            fc_layer = dropout(fc_layer, keep_prob=self.prob, is_training=is_training,
                               name='dropout_fc_%d' % i)
        fc_arm_layer = tf.matmul(fc_layer, weights['w_im_joint_fc%d' % (self.num_im_joint_fc_layer - 1)]) + weights[
            'b_im_joint_fc%d' % (self.num_im_joint_fc_layer - 1)]
        fc_grip_layer = tf.matmul(fc_layer, weights['w_im_grip_fc']) + weights['b_im_grip_fc']

        return fc_arm_layer, fc_grip_layer, fc_aux_layer

    def construct_model(self, input_tensors=None, prefix='Training_', dim_input=27, dim_output=7, network_config=None):
        """
        Construct the meta-learning graph.
        Args:
            input_tensors: tensors of input videos, if available
            prefix: indicate whether we are building training, validation or testing graph.
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            network_config: dictionary of network structure parameters
        Returns:
            a tuple of output tensors.
        """
        # Initialization parameters
        # a represent Training
        self.statea_image = tf.placeholder(tf.float32,
                                           name='statea_image')  # [Batch size, images length, width,  channel]
        self.statea_joint = tf.placeholder(tf.float32, name='statea_joint')  # [Batch size, Number of joints]
        self.actiona = tf.placeholder(tf.float32, name='actiona')  # [Batch size, Number of joints]
        # b represent Validation
        self.stateb_image = tf.placeholder(tf.float32, name='stateb_image')
        self.stateb_joint = tf.placeholder(tf.float32, name='stateb_joint')
        self.actionb = tf.placeholder(tf.float32, name='actionb')

        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            weights = self.weights = self.construct_weights()
            num_updates = self.num_updates
            lossa, lossb = [], []

            def cos_similary(a, b):
                a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1))
                b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1))
                a_b = tf.reduce_sum(tf.multiply(a, b), axis=-1)
                cosin = tf.divide(a_b, tf.multiply(a_norm, b_norm))
                return tf.acos(cosin, name='arccos')

            def batch_metalearn(inp):
                imagea, jointa, actiona, imageb, jointb, actionb = inp
                local_outputbs, local_lossesb, = [], []
                # Pre-update
                local_outputa = self.forward(imagea, jointa, weights)
                local_lossa = self.act_loss_eps * euclidean_loss_layer(local_outputa, actiona, multiplier=100,
                                                                       use_l1=True)
                # Cosine loss
                loss_cos = cos_similary(local_outputa, actiona)
                local_lossa += self.arccos_size * loss_cos

                # Compute fast gradients
                grads = tf.gradients(local_lossa, weights.values())
                gradients = dict(zip(weights.keys(), grads))
                # make fast gradient zero for weights with gradient None
                clip_min = self.clip_min
                clip_max = self.clip_max
                for key in gradients.keys():
                    if gradients[key] is None:
                        gradients[key] = tf.zeros_like(weights[key])
                    gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                fast_weights = dict(
                    zip(weights.keys(), [weights[key] - self.step_size * gradients[key] for key in weights.keys()]))

                # Post-update
                outputb = self.forward(imageb, jointb, fast_weights, meta_testing=True)
                local_outputbs.append(outputb)
                local_lossb = self.act_loss_eps * euclidean_loss_layer(outputb, actionb, multiplier=100,
                                                                       use_l1=True)
                # Cosine loss
                loss_cos = cos_similary(outputb, actionb)
                local_lossb += self.arccos_size * loss_cos

                local_lossesb.append(local_lossb)

                for i in range(num_updates - 1):
                    # Pre-update
                    outputa = self.forward(imagea, jointa, fast_weights)
                    local = self.act_loss_eps * euclidean_loss_layer(outputa, actiona, multiplier=100,
                                                                     use_l1=True)
                    loss_cos = cos_similary(local_outputa, actiona)
                    local += self.arccos_size * loss_cos

                    # Compute fast gradients
                    grads = tf.gradients(local, fast_weights.values())
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # make fast gradient zero for weights with gradient None
                    clip_min = self.clip_min
                    clip_max = self.clip_max
                    for key in gradients.keys():
                        if gradients[key] is None:
                            gradients[key] = tf.zeros_like(fast_weights[key])
                        gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                    fast_weights = dict(
                        zip(fast_weights.keys(),
                            [fast_weights[key] - self.step_size * gradients[key] for key in fast_weights.keys()]))

                    # Post-update
                    outputb = self.forward(imageb, jointb, fast_weights, meta_testing=True)
                    local_outputbs.append(outputb)
                    local_lossb = self.act_loss_eps * euclidean_loss_layer(outputb, actionb, multiplier=100,
                                                                           use_l1=True)
                    loss_cos = cos_similary(outputb, actionb)
                    local_lossb += self.arccos_size * loss_cos

                    local_lossesb.append(local_lossb)
                local_output = [local_outputa, local_outputbs, local_outputbs[-1], local_lossa, local_lossesb]
                return local_output

            # out_dtype=[tf.float32,[tf.float32]*num_updates]
        result = tf.map_fn(batch_metalearn, elems=(
            self.statea_image, self.statea_joint, self.actiona, self.stateb_image, self.stateb_joint, self.actionb),
                           dtype=[tf.float32, [tf.float32] * num_updates, tf.float32, tf.float32,
                                  [tf.float32] * num_updates])
        return result

    def construct_nn_model(self, input_tensors=None, prefix='Training_', dim_input=27, dim_output=7,
                           network_config=None):
        """
        Construct the learning graph.
        Args:
            input_tensors: tensors of input videos, if available
            prefix: indicate whether we are building training, validation or testing graph.
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            network_config: dictionary of network structure parameters
        Returns:
            a tuple of output tensors.
        """
        # Initialization parameters
        # a represent Training
        self.statea_image = tf.placeholder(tf.float32,
                                           name='statea_image')  # [Batch size, images length, width,  channel]
        self.statea_joint = tf.placeholder(tf.float32, name='statea_joint')  # [Batch size, Number of joints]
        self.actiona = tf.placeholder(tf.float32, name='actiona')  # [Batch size, Number of joints]
        self.object = tf.placeholder(tf.float32, name='object')  # [Batch size, Number of object x y z]

        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            weights = self.weights = self.construct_weights()

            def cos_similary(a, b):
                a_norm = tf.sqrt(tf.reduce_sum(tf.square(a)))
                b_norm = tf.sqrt(tf.reduce_sum(tf.square(b)))
                a_b = tf.reduce_sum(tf.multiply(a, b))
                cosin = tf.divide(a_b, tf.multiply(a_norm, b_norm))
                return tf.acos(cosin, name='arccos')

            def batch_metalearn(inp):
                imagea, jointa, actiona, object = inp
                arm_action = tf.concat([tf.expand_dims(actiona[:, 0], -1), actiona[:, 2:]], 1)
                grip_action = tf.expand_dims(actiona[:, 1], -1)
                # Pre-update
                local_outputa, local_outputa_grip, local_object_output = self.forward(imagea, jointa, weights)
                local_lossa = self.act_loss_eps * euclidean_loss_layer(local_outputa, arm_action, multiplier=500,
                                                                       use_l1=True)
                local_grip_loss = self.act_loss_eps * euclidean_loss_layer(local_outputa_grip, grip_action,
                                                                           multiplier=500,
                                                                           use_l1=True)
                local_lossa += local_grip_loss

                local_object = self.act_loss_eps * euclidean_loss_layer(local_object_output, object,
                                                                        multiplier=100,
                                                                        use_l1=True)

                local_lossa += local_object

                # Cosine loss
                loss_cos = cos_similary(local_outputa, arm_action)
                local_lossa += self.arccos_size * loss_cos

                return [local_lossa, local_outputa, local_outputa_grip]

            # out_dtype=[tf.float32,[tf.float32]*num_updates]
        result = batch_metalearn([self.statea_image, self.statea_joint, self.actiona, self.object])

        return result
