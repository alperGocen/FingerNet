import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications as app
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
import cv2
import io
import matplotlib.pyplot as plt


# Conv-Block 1
model = Sequential()
model.add(Conv2D(input_shape=(256,256,1), filters=40, strides=1, activation="relu", kernel_size=4, padding="same"))
model.add(Conv2D(filters=40, strides=1, activation="relu", kernel_size=4, padding="same"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
##########

# Pooling-1
model.add(MaxPool2D(pool_size=(2,2)))
##########

# Conv-Block 2
model.add(Conv2D(filters=40, strides=1, activation="relu", kernel_size=3,padding="same"))
model.add(Conv2D(filters=60, strides=1, activation="relu", kernel_size=3, padding="same"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
###########

# Pooling-2
model.add(MaxPool2D(pool_size=(2,2)))
###########

# Conv-Block 3
model.add(Conv2D(filters=60, strides=1, activation="relu", kernel_size=3, padding="same"))
model.add(Conv2D(filters=80, strides=1, activation="relu", kernel_size=3, padding="same"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
###########

# Pooling-2
model.add(MaxPool2D(pool_size=(2,2)))
###########

# Conv-Block 4
model.add(Conv2D(filters=80, strides=2, activation="relu", kernel_size=2, padding="same"))
model.add(Conv2D(filters=120, strides=2, activation="relu", kernel_size=2, padding="same"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
###########

# Conv-Block 5
model.add(Conv2D(filters=120, strides=2, activation="relu", kernel_size=2, padding="same"))
model.add(Conv2D(filters=160, strides=2, activation="relu", kernel_size=2, padding="same"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
###########



model.summary()








