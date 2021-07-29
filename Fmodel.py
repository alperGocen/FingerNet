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
from keras.preprocessing.image import ImageDataGenerator
import cv2
import io
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
from keras import backend as K
from math import log
from tensorflow.keras.preprocessing.image import random_rotation
from keras import backend as K

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import os
import cv2
import time
import numpy as np
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

%cd /content/drive/My\ Drive/FVC_2002/FVC2002

from google.colab import drive
drive.mount('/content/drive')

image_size = (160, 160)  # the size of resized images
image_channels = 1

out_feature_length = 320  # the length of feature

inputs = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], image_channels])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)


def conv_layer(prev_layer, num_units, kernal_size, strides_size, is_training, name):
    conv_layer = tf.layers.conv2d(prev_layer, num_units, kernal_size, strides_size, padding='same',
                                  use_bias=True, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.elu(conv_layer, name=name)
    return conv_layer


def max_pool_2x2(prev_layer, name):
    return tf.layers.max_pooling2d(prev_layer, pool_size=[2, 2], strides=2, padding='same', name=name)


def inference_test(images, phase, keep_prob):
    # conv1
    with tf.variable_scope('conv1') as scope:
        conv1_1 = conv_layer(images, 20, 4, 1, phase, name=scope.name)
        conv1_2 = conv_layer(conv1_1, 40, 4, 1, phase, name=scope.name)

    # pool1
    with tf.variable_scope('pooling1') as scope:
        pool1 = max_pool_2x2(conv1_2, name=scope.name)

    # conv2
    with tf.variable_scope('conv2') as scope:
        conv2_1 = conv_layer(pool1, 40, 3, 1, phase, name=scope.name)
        conv2_2 = conv_layer(conv2_1, 80, 3, 1, phase, name=scope.name)

    # pool2
    with tf.variable_scope('pooling2') as scope:
        pool2 = max_pool_2x2(conv2_2, name=scope.name)

    # conv3
    with tf.variable_scope('conv3') as scope:
        conv3_1 = conv_layer(pool2, 80, 3, 1, phase, name=scope.name)
        conv3_2 = conv_layer(conv3_1, 120, 3, 1, phase, name=scope.name)

    # pool3
    with tf.variable_scope('pooling3') as scope:
        pool3 = max_pool_2x2(conv3_2, name=scope.name)

    # conv4
    with tf.variable_scope('conv4') as scope:
        conv4_1 = conv_layer(pool3, 120, 2, 2, phase, name=scope.name)
        conv4_2 = conv_layer(conv4_1, 160, 2, 1, phase, name=scope.name)

    # conv5
    with tf.variable_scope('conv5') as scope:
        conv5_1 = conv_layer(conv4_2, 160, 2, 2, phase, name=scope.name)
        conv5_2 = conv_layer(conv5_1, 320, 2, 1, phase, name=scope.name)


    # local3
    with tf.variable_scope('fc_1') as scope:
        reshape = tf.reshape(conv5_2, shape=[-1, 5 * 5 * 320])
        weights = tf.get_variable('weights',
                                  shape=[5 * 5 * 320, out_feature_length],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[out_feature_length, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc_1 = tf.matmul(reshape, weights) + biases
        fc_1 = tf.layers.batch_normalization(fc_1, training=phase)

        local3 = tf.nn.elu(fc_1, name=scope.name)

    # dropout
    with tf.variable_scope('dropout') as scope:
        dropout_layer = tf.nn.dropout(local3, keep_prob, name=scope.name)

    return dropout_layer

def inference_test_nn(nn_images):
    with tf.variable_scope('dense1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[out_feature_length*2, out_feature_length*2],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[out_feature_length*2, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc_1 = tf.matmul(nn_images, weights) + biases
        local3 = tf.nn.relu(fc_1, name=scope.name)
        regularizer_1 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

    with tf.variable_scope('dense2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[out_feature_length*2, 2],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[2, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        logits = tf.matmul(local3, weights) + biases
        regularizer_2 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

    regularizers = (regularizer_1 + regularizer_2)

    return logits, regularizers

def binary_classifier_model():
  model = Sequential()

  model.add(Dense(input_shape=(640, ), units=640, activation='relu'))
  model.add(Dense(640, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer = 'Adam', metrics=['accuracy'])

  return model

def get_class_path(root_path, test_file):
    relative_image_path = []
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dirname = line.strip()
            absolute_dirname = os.path.join(root_path, dirname)
            if not os.path.isdir(absolute_dirname):
                raise IOError('{} is not a directory!'.format(absolute_dirname))
            for img_name in os.listdir(absolute_dirname):
                relative_path = os.path.join(dirname, img_name)
                relative_image_path.append(relative_path)
    return relative_image_path


def read_image(root_path, test_file, resize=None):
    relative_image_path = get_class_path(root_path, test_file)
    width = image_size[0]
    height = image_size[1]
    temp = np.zeros([len(relative_image_path), width, height, 1], np.float32)

    for i, img_path in enumerate(relative_image_path):
        im = cv2.imread(os.path.join(root_path, img_path), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (width, height))
        temp[i, :, :, 0] = im
    temp = np.asarray(temp, np.float32)
    return list(temp), relative_image_path


def extract_feature(session, layer, _LATEST_CHECKPOINT, root_path, test_file, once_read_imagenum=1):
    ims, relative_image_path = read_image(root_path, test_file)

    image_num = len(relative_image_path)
    total_iters = int(np.ceil(image_num / once_read_imagenum)) 

    all_feature_matrix = np.zeros((image_num, out_feature_length), dtype=np.float32)  

    print("begin to extract feature")
    saver = tf.train.Saver()
    saver.restore(session, _LATEST_CHECKPOINT)
    for iter in range(total_iters):
        start = iter * once_read_imagenum
        end = min(start + once_read_imagenum, image_num)
        batch_ims = ims[start:end]

        batch_image_features = session.run(layer, feed_dict={inputs: np.array(batch_ims), keep_prob: 1.0,
                                                             is_training: False})
        all_feature_matrix[start:end] = batch_image_features
        print('process {} images'.format(end))

    print("extract features done")
    return all_feature_matrix, relative_image_path


def get_feature_file(session, layer, _LATEST_CHECKPOINT, root_path, test_file, out_dir):
    all_feature_matrix, relative_image_path = extract_feature(session, layer, _LATEST_CHECKPOINT, root_path, test_file)
    image_num = all_feature_matrix.shape[0]
    assert len(relative_image_path) == image_num

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('match result:\n')
    for i in range(image_num):
        class_name = os.path.dirname(relative_image_path[i])
        save_dir = os.path.join(out_dir, class_name)
        if not os.path.isdir(save_dir):
            print('Create directory: {}'.format(save_dir))
            os.makedirs(save_dir)
        name_prefix = os.path.splitext(os.path.basename(relative_image_path[i]))[0]
        file_name = os.path.join(save_dir, name_prefix + '.fea')
        all_feature_matrix[i].tofile(file_name)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('process {} images!'.format(image_num))

def gene_feature_dict(_fea_root_path): # constructs a dictionary of names and corresponding features
    _name2feature = {}
    fea_class_files = os.listdir(_fea_root_path)
    for line in fea_class_files:
        class_path = os.path.join(_fea_root_path, line)
        for fea_file in os.listdir(class_path):
            file_path = os.path.join(class_path, fea_file)
            name_prefix = os.path.splitext(fea_file)[0]
            feature = np.fromfile(file_path, dtype=np.float32)
            _name2feature[name_prefix] = feature
            
    return _name2feature

def get_idy_file(root_path, _fea_root_path, nn_input, _LATEST_CHECKPOINT, test_file, out_dir):
    relative_image_path = get_class_path(root_path, test_file)
    name2feature = gene_feature_dict(_fea_root_path)
    
    filename_list = []
    for line in relative_image_path:
        _name_prefix = os.path.splitext(os.path.basename(line))[0]
        filename_list.append(_name_prefix)
    
    image_num = len(relative_image_path)
    assert len(filename_list) == image_num

    if not os.path.isdir(out_dir):
        print('Create directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    temp_files = os.listdir(out_dir)
    NN_layer, regularizers = inference_test_nn(nn_input)

    session = tf.Session()
    print(tf.global_variables())
    name_to_var_map = {var.op.name: var for var in tf.global_variables()}
    nn_saver = tf.train.Saver(name_to_var_map)

    nn_saver.restore(session, _LATEST_CHECKPOINT)

    for i in range(image_num):
        name_prefix = filename_list[i]
        temp_name = name_prefix + '.txt'
        if temp_name not in temp_files:
            input_feature_matrix = [np.concatenate((name2feature[filename_list[i]], name2feature[filename_list[j]])) for j in range(image_num)]
            input_feature_matrix = np.array(input_feature_matrix).reshape((image_num, 2 * out_feature_length))

            out = session.run(NN_layer, {nn_input: input_feature_matrix})

            predicts = tf.nn.softmax(tf.cast(out, tf.float32))

            predict_matrix = np.float32(session.run(predicts))

            assert image_num == predict_matrix.shape[0]

            file_name = os.path.join(out_dir, temp_name)
            with open(file_name, 'w') as file: 
                for j in range(image_num):
                    if i != j:
                        match_image_name = filename_list[j]
                        if os.path.dirname(relative_image_path[i]) == os.path.dirname(relative_image_path[j]):
                            label = 1
                        else:
                            label = 0
                        out_line = "{:s}\t{:d}\t{:.5f}".format(match_image_name, label, predict_matrix[j][-1])
                        file.writelines(out_line + '\n')
                    else:
                        pass

            print('generate {} txt file'.format(i))
        else:
            print('file {0} exits'.format(temp_name))
    session.close()
    print('process {} images!'.format(image_num))


def prepare_feature_set(root_path, _fea_root_path, test_file):
    relative_image_path = get_class_path(root_path, test_file)
    name2feature = gene_feature_dict(_fea_root_path)
    
    filename_list = []
    for line in relative_image_path:
        _name_prefix = os.path.splitext(os.path.basename(line))[0]
        filename_list.append(_name_prefix)
    
    image_num = len(relative_image_path)
    assert len(filename_list) == image_num

    all_input_matrix = np.array([])

    for i in range(image_num):
        name_prefix = filename_list[i]
        temp_name = name_prefix + '.txt'

        input_feature_matrix = [np.concatenate((name2feature[filename_list[i]], name2feature[filename_list[j]])) for j in range(image_num)]
        input_feature_matrix = [np.concatenate((input_feature_matrix[i], [1] if filename_list[i].split('_')[0] == filename_list[j].split('_')[0] else [0]), axis=0) for j in range(image_num)]
        input_feature_matrix = np.array(input_feature_matrix).reshape((image_num, 2 * out_feature_length + 1))

        if all_input_matrix.size == 0:
          all_input_matrix = input_feature_matrix

        all_input_matrix = np.vstack((all_input_matrix, input_feature_matrix))

    return all_input_matrix


if __name__ == '__main__':
    image_root_dir = 'Dbs/Db1/data/Db1_a'     # the path of images

    test_file = 'Dbs/Db1/data/test/test.txt'       # the test file

    CHECKPOINT_PATH = 'Dbs/Db1/data/new_check'       # the path of trained model
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)

    #layer = inference_test(inputs, is_training, keep_prob)

    #Feature Extraction ------
    #out_dir = 'Dbs/Db1/data/features'      # the path for feature files

    #with tf.Session() as sess:
    #    get_feature_file(sess, layer, LATEST_CHECKPOINT, image_root_dir, test_file, out_dir)
    
    # NN-Classifier Training
    feature_root_path = 'Dbs/Db1/data/features'
    feature_input = tf.placeholder(tf.float32, [None, out_feature_length*2])

    save_idy_file_dir = 'Dbs/Db1/data/matching'

    get_idy_file(image_root_dir, feature_root_path, feature_input, LATEST_CHECKPOINT, test_file, save_idy_file_dir)



def add_conv_block(model, num_of_inputs, strides, kernel_size):
  model.add(Conv2D(input_shape=(256,256,3), filters=num_of_inputs, strides=strides, kernel_size=kernel_size, padding="same"))
  model.add(BatchNormalization())
  model.add(ELU(alpha=1.0))


def add_max_pool_block(model):
  model.add(MaxPool2D(pool_size=(2,2), strides=2, padding="same"))


def train_model(model):
  train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=20)

  validation_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=20)

  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        'Dbs/Db1/data/Db1_a',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 256x256
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical')
  
  validation_generator = validation_datagen.flow_from_directory(
        'Dbs/Db1/data/Db1_av',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 256x256
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical')
  
  model.fit(
        train_generator,
        validation_data = validation_generator,
        verbose=1,
        steps_per_epoch= 595,
        epochs=50)
  
  model.save_weights('first_try.h5')

# coding: utf-8
 
import tensorflow as tf
 
def _center_loss_func(features, labels, alpha, num_classes):
    with tf.compat.v1.variable_scope('center_loss',reuse = tf.compat.v1.AUTO_REUSE):  
                 # Get the dimension of the feature, such as 256 dimensions
        len_features = features.get_shape()[1]
                 # Create a Variable, shape [num_classes, len_features], to store the sample center of the entire network,
                 # Set trainable=False because the center of the sample is not updated by the gradient
        centers = tf.compat.v1.get_variable('centers', [num_classes, len_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
                 # Expand the label into one-dimensional, if the input is already one-dimensional, the action is actually unnecessary
        labels = tf.reshape(labels, [-1])
                 # According to the sample label, get the center value of each sample in the mini-batch
        labels = tf.compat.v1.to_int32(labels)
        centers_batch = tf.gather(centers, labels)
                 # print('centers_batch:',centers_batch.get_shape().as_list())
                 # Calculate loss
        loss = tf.nn.l2_loss(features - centers_batch)
                 # The difference between the eigenvalues ​​of the current mini-batch and their corresponding center values
        diff = centers_batch - features
                 # Get the number of occurrences of samples of the same category in mini-batch. For the principle, please refer to the original formula (4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx) ###Use a one-dimensional index array to extract the vector corresponding to the index in the tensor
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        centers_update_op = tf.compat.v1.scatter_sub(centers, labels, diff)
        '''
        scatter_sub(ref,indices,updates)
                 ref: a variable Tensor; must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half; it should come from a Variable node.
                 indices: a Tensor; must be one of the following types: int32, int64; an index tensor into the first dimension of ref.
                 updates: A Tensor. Must have the same type as ref. Subtract the tensor of updated values ​​from ref.
        '''
                 ###centers_update_op Update center op, need run to update center
        return loss

def get_center_loss(alpha, num_classes):  
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, 100])
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return tf.compat.v1.to_float(cce(y_true, y_pred)) + 0.5 * _center_loss_func(y_pred, y_true, alpha, num_classes)
    return center_loss

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
  #add_max_pool_block(model)
  ###########

  # Conv-Block 4
  #add_conv_block(model, 120, 1, 2)
  #add_conv_block(model, 160, 1, 2)
  ###########

  # Conv-Block 5
  #add_conv_block(model, 160, 1, 2)
  #add_conv_block(model, 320, 1, 2)
  ###########

  model.add(Flatten())
  model.add(Dense(120, activation='relu'))
  model.add(Dense(100, activation='softmax'))

  optimizer = keras.optimizers.Adam(learning_rate=0.00001)

  center_loss = get_center_loss(0.5, 100)

  total_loss = center_loss

  model.compile(loss=total_loss,
              optimizer= optimizer,
              metrics=['accuracy'])

  return model

from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze four convolution blocks
for layer in model.layers[:15]:
    layer.trainable = False

print("alper")

model.layers.pop()

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

x = model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(100, activation='softmax')(x) # Softmax for multiclass
model = Model(inputs=model.input, outputs=x)

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=5e-5), metrics=["accuracy"])

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 1)

nn_classifier = binary_classifier_model()

model.summary()

train_model(model)




