import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Reshape, Softmax, Conv2DTranspose
from keras.regularizers import l2

lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

IMG_SIZE = (512, 512, 1)

model = Sequential()
model.add(Conv2D(32, (8, 8), input_shape=IMG_SIZE, padding="same", activation=lrelu, kernel_regularizer=l2(5e-4), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Conv2D(filters=64, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Conv2D(filters=32, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=256, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=32, kernel_size=(1, 1), padding="same", activation=lrelu, kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.25))

model.add(Conv2DTranspose(11, kernel_size=5, strides=1, padding="same", activation=lrelu))
model.add(Reshape((32, 32, 11)))
model.add(Softmax(3))
