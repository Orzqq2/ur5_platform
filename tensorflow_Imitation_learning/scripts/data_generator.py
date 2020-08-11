""" Code for loading data and generating data batches during training """
from __future__ import division

import copy
import logging
import os
import glob
import tempfile
import pickle
import h5py
from datetime import datetime
from collections import OrderedDict

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from random import shuffle

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, update_batch_size=50, test_batch_size=20, meta_batch_size=3):
        # Hyperparameters
        self.update_batch_size = update_batch_size
        self.test_batch_size = test_batch_size
        self.meta_batch_size = meta_batch_size
        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        self.extract_supervised_data()

    def extract_supervised_data(self):
        """
            Load the states and actions of the demos into memory.
            Args:
                demo_file: list of demo files where each file contains expert's states and actions of one task.
        """
        self.demo_image, self.demo_joint, self.demo_action, self.demo_object = np.array([]), np.array([]), np.array(
            []), np.array([])
        for i in range(2, 3):
            file_dir = 'data/storage_date_100_130_c4_' + str(i) + '.hdf5'
            demo_file = h5py.File(file_dir, 'r')
            # [step, Single step dimension]
            # demo_image: [step, Width, Height, Channel]
            for key in demo_file.keys():
                print key
                if 'velocity' in key:
                    if len(self.demo_action) == 0:
                        self.demo_action = demo_file[key]
                    else:
                        self.demo_action = np.append(self.demo_action, demo_file[key], axis=0)
                elif 'position' in key:
                    if len(self.demo_joint) == 0:
                        self.demo_joint = demo_file[key]
                    else:
                        self.demo_joint = np.append(self.demo_joint, demo_file[key], axis=0)
                elif 'image' in key:
                    if len(self.demo_image) == 0:
                        self.demo_image = demo_file[key]
                    else:
                        self.demo_image = np.append(self.demo_image, demo_file[key], axis=0)
                elif 'object' in key:
                    if len(self.demo_object) == 0:
                        self.demo_object = demo_file[key]
                    else:
                        self.demo_object = np.append(self.demo_object, demo_file[key], axis=0)
            demo_file.close()
        print 'data finish'

    def generate_data_batch(self, itr):
        # Collect samples
        length = len(self.demo_image) - (self.update_batch_size + self.test_batch_size)
        data_train_batch_image = [
            self.demo_image[
            (itr + i) % (length - 5):(itr + i) % (length - 5) + (self.update_batch_size + self.test_batch_size)]
            for i in range(self.meta_batch_size)]
        data_train_batch_joint = [
            self.demo_joint[
            (itr + i) % (length - 5):(itr + i) % (length - 5) + (self.update_batch_size + self.test_batch_size)]
            for i in
            range(self.meta_batch_size)]
        data_train_batch_action = [
            self.demo_action[
            (itr + i) % (length - 5):(itr + i) % (length - 5) + (self.update_batch_size + self.test_batch_size)]
            for i in
            range(self.meta_batch_size)]

        return np.array(data_train_batch_image), np.array(data_train_batch_joint), np.array(data_train_batch_action)

    def generate_data_batch_nn(self, itr):
        # Collect samples
        length = len(self.demo_image) - (self.update_batch_size + self.test_batch_size)
        data_train_batch_image = self.demo_image[itr % (length - 5):itr % (length - 5) + (
                self.update_batch_size + self.test_batch_size)]

        data_train_batch_joint = self.demo_joint[itr % (length - 5):itr % (length - 5) + (
                self.update_batch_size + self.test_batch_size)]

        data_train_batch_action = self.demo_action[itr % (length - 5):itr % (length - 5) + (
                self.update_batch_size + self.test_batch_size)]

        data_train_batch_object = self.demo_object[itr % (length - 5):itr % (length - 5) + (
                self.update_batch_size + self.test_batch_size)]

        return np.array(data_train_batch_image), np.array(data_train_batch_joint), np.array(
            data_train_batch_action), np.array(data_train_batch_object)
