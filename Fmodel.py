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
from keras.preprocessing.image import ImageDataGenerator
import cv2
import io
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

%cd /content/drive/My\ Drive/FVC_2002/FVC2002

from google.colab import drive
drive.mount('/content/drive')

def add_conv_block(model, num_of_inputs, strides, kernel_size):
  model.add(Conv2D(input_shape=(256,256,1), filters=num_of_inputs, strides=strides, activation="relu", kernel_size=kernel_size, padding="same"))
  model.add(BatchNormalization())
  model.add(ELU(alpha=1.0))


def add_max_pool_block(model):
  model.add(MaxPool2D(pool_size=(2,2), strides=2, padding="same"))


def train_model(model):
  train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        'Dbs/Db1_a',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 256x256
        batch_size=32,
        class_mode='binary')
  
  model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // 32,
        epochs=50)
  
  model.save_weights('first_try.h5')

def custom_model():
  # Conv-Block 1
  model = Sequential()
  add_conv_block(model, 20, 1, 4)
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
  model.add(Dense(320, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))

  model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

  return model


model = custom_model()
model.summary()

train_model(model)