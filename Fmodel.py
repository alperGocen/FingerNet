import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications as app
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.models import Model
import cv2
import io
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

!pip3 uninstall keras-nightly
!pip3 uninstall -y tensorflow
!pip3 install keras==2.1.6
!pip3 install tensorflow==1.15.0
!pip3 install h5py==2.10.0

import functools
import os
import cv2
import time
import numpy as np
import tensorflow as tfs
import tensorflow.compat.v1 as tf
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import tensorboard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from math import log
from tensorflow.keras.preprocessing.image import random_rotation
from keras import backend as K
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


tf.disable_v2_behavior()

%cd /content/drive/My\ Drive/FVC_2002/FVC2002

from google.colab import drive
drive.mount('/content/drive')


#Block definitions
def add_conv_block(model, num_of_inputs, strides, kernel_size):
  model.add(Conv2D(filters=num_of_inputs, strides=strides, kernel_size=kernel_size, padding="same"))
  model.add(BatchNormalization())
  model.add(ELU())

def add_max_pool_block(model):
  model.add(MaxPool2D(pool_size=(2,2), strides=2, padding="same"))

batch_size = 16


# Center loss from the paper in link https://ydwen.github.io/papers/WenECCV16.pdf
def _center_loss_func(features, labels, alpha, num_classes, centers, feature_dim): 
        assert feature_dim == features.get_shape()[1]
        pdb.set_trace() 
        labels = K.reshape(labels, [-1])
        labels = tf.to_int32(labels)
        features = tf.to_int32(features)
        centers_batch = tf.gather(centers, labels)
        diff = (1 - alpha) * (centers_batch - features)
        centers = tf.scatter_sub(centers, labels, diff)
        loss = tf.reduce_mean(K.square(features - centers_batch))

        return loss

def get_center_loss(alpha, num_classes, feature_dim): 
    # Each output layer use one independed center: scope/centers
    centers = np.zeros([num_classes, feature_dim])
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha, 
                                 num_classes, centers, feature_dim)
    return center_loss


# Training of the model
def train_model(model):
    # Rotation in a range of +-15 pixels and shifting in +-20 pixels
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=40)

    validation_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=40)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'Dbs/Db1/data/Db1_a',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 256x256
        batch_size=16,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
    'Dbs/Db1/data/Db1_av',  # this is the target directory
    target_size=(256, 256),  # all images will be resized to 256x256
    batch_size=16,
    color_mode='grayscale',
    class_mode='categorical')

    model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    verbose=1,
    steps_per_epoch= 594,
    epochs=75)


    model.save_weights('first_try.h5')


# Definition of Fingernet model
def custom_model():
    # Conv-Block 1
    model = Sequential()
    model.add(Conv2D(batch_input_shape=(16, 256, 256, 1), filters=20, strides=1, kernel_size=4, padding="same"))
    model.add(BatchNormalization())
    model.add(ELU())

    add_conv_block(model, 40, 1, 4)
    ##########

    # Pooling-1
    add_max_pool_block(model)
    ##########

    # Conv-Block 2
    add_conv_block(model, 40, 1, 3)
    add_conv_block(model, 80, 1, 3)
    ###########

    # Pooling-2
    add_max_pool_block(model)
    ###########

    # Conv-Block 3
    add_conv_block(model, 80, 1, 3)
    add_conv_block(model, 120, 1, 3)

    ###########

    # Pooling-3
    add_max_pool_block(model)
    ###########

    # Conv-Block 4
    add_conv_block(model, 120, 1, 2)
    add_conv_block(model, 160, 1, 2)
    ###########

    # Conv-Block 5
    add_conv_block(model, 160, 1, 2)
    add_conv_block(model, 320, 1, 2)
    ###########

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(99, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.00001)

    model.summary()

    center_loss = get_center_loss(0.5, 99, 99)

    model.compile(loss=center_loss,
              optimizer= optimizer,
              metrics=['accuracy'])

    return model



# Call necessary methods to start processing
model = custom_model()

model.summary()

train_model(model)



