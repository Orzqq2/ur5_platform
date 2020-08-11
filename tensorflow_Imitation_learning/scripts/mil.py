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

    def __init__(self):

        # MIL hyperparams
        self.activation_fn = tf.nn.relu
        self.num_updates = 5
        self.clip_min = 10
        self.clip_max = -10
        self.step_size = 1e-3
        self.meta_batch_size = 3
        self.meta_lr = 0.001
        self.act_loss_eps = 0.9
        self.prob = 0.6

    def init_network(self, graph, prefix='Training_'):
        """ Helper method to initialize the tf networks used """
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
                self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses2[self.num_updates - 1])

    def construct_weights(self, dim_input=27, dim_output=7):
        """ Construct weights for the network. """
        weights = {}
        # Define the number of base layers, filter size, and step length
        # Image Conv Processing
        n_conv_layers = self.n_conv_layers = 4  # 5-layer convolution
        filter_sizes = [3] * n_conv_layers
        strides = [1, 2, 2, 1]
        im_height = 100
        im_width = 130
        im_num_channels = 4
        fan_in = im_num_channels
        num_filters = [32, 64, 32, 16]
        for i in range(n_conv_layers):
            weights['wc%d' % (i + 1)] = init_conv_weights_snn(
                [filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]],
                name='wc%d' % (i + 1))
            weights['bc%d' % (i + 1)] = init_bias([num_filters[i]], name='bc%d' % (i + 1))
            fan_in = num_filters[i]
        self.shape_of_out_conv = [im_height // (2 ** 4) + 1, im_width // (2 ** 4) + 1, num_filters[-1]]

        # Image FC Processing
        fc_in_shape = self.fc_in_shape = int(reduce(lambda x, y: x * y, self.shape_of_out_conv))
        print fc_in_shape
        self.num_conv_fc_layer = 2
        num_fc_filters = [512, 64]
        for i in range(self.num_conv_fc_layer):
            weights['w_1d_conv%d' % i] = init_fc_weights_snn([fc_in_shape, num_fc_filters[i]], name='w_1d_conv%d' % i)
            weights['b_1d_conv%d' % i] = init_bias(num_fc_filters[i], name='b_1d_conv%d' % i)
            fc_in_shape = num_fc_filters[i]

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
        num_im_joint_filters = [64, 7]
        for i in range(num_im_joint_fc_layer):
            weights['w_im_joint_fc%d' % i] = init_fc_weights_snn([im_joint_fc_in, num_im_joint_filters[i]],
                                                                 name='w_im_joint_fc%d' % i)
            weights['b_im_joint_fc%d' % i] = init_bias(num_im_joint_filters[i], name='b_im_joint_fc%d' % i)
            im_joint_fc_in = num_im_joint_filters[i]
        return weights

    # def construct_fc_weights(self, dim_input=27, dim_output=7, network_config=None):

    def forward(self, image_input, state_input, weights, meta_testing=False, is_training=True, testing=False,
                network_config=None):
        """ Perform the forward pass. """
        strides = [1, 2, 2, 1]
        # Conv process
        conv_layer = image_input
        for i in range(self.n_conv_layers):
            conv_layer = dropout(norm(
                conv2d(img=conv_layer, w=weights['wc%d' % (i + 1)], b=weights['bc%d' % (i + 1)], strides=strides,
                       is_dilated=False), decay=0.9, id=i, is_training=is_training, activation_fn=tf.nn.relu),
                keep_prob=self.prob, is_training=is_training, name='dropout_conv_%d' % (i + 1))

        # Conv fc process, output shape is [-1, 64]
        conv_fc_ouput = tf.reshape(conv_layer, [-1, self.fc_in_shape])
        for i in range(self.num_conv_fc_layer):
            conv_fc_ouput = tf.matmul(conv_fc_ouput, weights['w_1d_conv%d' % i]) + weights['b_1d_conv%d' % i]
            conv_fc_ouput = self.activation_fn(conv_fc_ouput)
            conv_fc_ouput = dropout(conv_fc_ouput, keep_prob=self.prob, is_training=is_training,
                                    name='dropout_conv_fc_%d' % i)

        # Joint fc process, output shape is [-1, 64]
        joint_layer = state_input  # [-1, 7]
        for i in range(self.num_joint_fc_layer):
            joint_layer = tf.matmul(joint_layer, weights['w_joint_fc%d' % i]) + weights['b_joint_fc%d' % i]
            joint_layer = self.activation_fn(joint_layer)
            joint_layer = dropout(joint_layer, keep_prob=self.prob, is_training=is_training,
                                  name='dropout_joint_fc_%d' % i)

        fc_layer = tf.concat([conv_fc_ouput, joint_layer], 1)  # [-1, 128]
        # FC process, output is [-1, 7] (action)
        for i in range(self.num_im_joint_fc_layer):
            fc_layer = tf.matmul(fc_layer, weights['w_im_joint_fc%d' % i]) + weights['b_im_joint_fc%d' % i]
            if i != self.num_im_joint_fc_layer - 1:
                fc_layer = self.activation_fn(fc_layer)
                fc_layer = dropout(fc_layer, keep_prob=self.prob, is_training=is_training,
                                   name='dropout_fc_%d' % i)
            else:
                fc_layer = fc_layer

        return fc_layer

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

            def batch_metalearn(inp):
                imagea, jointa, actiona, imageb, jointb, actionb = inp
                local_outputbs, local_lossesb, = [], []
                # Pre-update
                local_outputa = self.forward(imagea, jointa, weights)
                local_lossa = self.act_loss_eps * euclidean_loss_layer(local_outputa, actiona, multiplier=100,
                                                                       use_l1=True)

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
                local_lossesb.append(local_lossb)

                for i in range(num_updates - 1):
                    # Pre-update
                    outputa = self.forward(imagea, jointa, fast_weights)
                    local = self.act_loss_eps * euclidean_loss_layer(outputa, actiona, multiplier=100,
                                                                     use_l1=True)
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
                    local_lossesb.append(local_lossb)
                local_output = [local_outputa, local_outputbs, local_outputbs[-1], local_lossa, local_lossesb]
                return local_output

            # out_dtype=[tf.float32,[tf.float32]*num_updates]
        result = tf.map_fn(batch_metalearn, elems=(
            self.statea_image, self.statea_joint, self.actiona, self.stateb_image, self.stateb_joint, self.actionb),
                           dtype=[tf.float32, [tf.float32] * num_updates, tf.float32, tf.float32,
                                  [tf.float32] * num_updates])
        return result
