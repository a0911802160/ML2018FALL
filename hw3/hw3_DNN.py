import numpy as np
import pandas as pd
import sys
import math
import csv
import os

import tensorflow as tf
import keras.backend as k
from keras.backend.tensorflow_backend import set_session


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback, ReduceLROnPlateau
from sklearn import cross_validation, ensemble, preprocessing, metrics
from keras.utils import np_utils
# from keras.utils.vis_utils import plot_model as plot


data = pd.read_csv(sys.argv[1], delimiter=',')
x = np.array(data.iloc[:, 1])
train_y = np.array(data.iloc[:, 0])

train_x = []
for idx in range(len(x)):
    train_x.append(np.reshape(
        np.array(x[idx].split(' '), dtype=np.float64), (1, 48, 48)))
train_x = np.array(train_x)
train_x = train_x/255


train_y = np_utils.to_categorical(train_y, num_classes=7)

np.random.seed(321)
val_len = int(len(train_x)*.1)
order = np.arange(len(train_x))
np.random.shuffle(order)
val_x = train_x[order[:val_len]]
val_y = train_y[order[:val_len]]

data_generator_train = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, data_format='channels_first')


data_generator_train.fit(train_x)

model = Sequential()

model.add(Flatten(input_shape=(1, 48, 48)))


# layer_2

model.add(Dense(1024))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dropout(0.2))

###############
# layer_3

model.add(Dense(512))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dropout(0.2))

###############
# layer_4

model.add(Dense(256))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization(axis=1, momentum=0.99,
                             epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))

model.add(Dropout(0.2))

###############

model.add(Dense(7, activation='softmax'))

model.summary()
os.system('pause')
plot(model, to_file='DNN_model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-2), metrics=['accuracy'])

checkpoint = ModelCheckpoint('ep{epoch:03d}-val_acc{val_acc:.3f}.h5',
                             monitor='val_acc', save_weights_only=False, save_best_only=True, period=5)

model.fit_generator(data_generator_train.flow(
    train_x, train_y, batch_size=128), validation_data=(val_x, val_y), validation_steps=val_len/128, epochs=75,
    initial_epoch=0, callbacks=[checkpoint])


reduce_lr = ReduceLROnPlateau(
    monitor='val_acc', factor=0.5, patience=20, verbose=1)
early_stopping = EarlyStopping(
    monitor='val_acc', min_delta=0, patience=100, verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3), metrics=['accuracy'])

model.fit_generator(data_generator_train.flow(
    train_x, train_y, batch_size=128), validation_data=(val_x, val_y), epochs=300,
    initial_epoch=75, callbacks=[reduce_lr, early_stopping])

model.save('DNN_model.h5')
