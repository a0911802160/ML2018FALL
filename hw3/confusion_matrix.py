from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import csv
import math
from keras.models import load_model
from keras import backend as k
import numpy as np
import pandas as pd
import sys
import math
import csv
import itertools
import tensorflow as tf
import keras.backend as k
from keras.backend.tensorflow_backend import set_session

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

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

# from sklearn official site


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


data = pd.read_csv(sys.argv[1], delimiter=',')
x = np.array(data.iloc[:, 1])
train_y = np.array(data.iloc[:, 0])

train_x = []
for idx in range(len(x)):
    train_x.append(np.reshape(
        np.array(x[idx].split(' '), dtype=np.float64), (1, 48, 48)))
train_x = np.array(train_x)

train_x = train_x/255

model = load_model('best.h5')
label = model.predict(train_x)
label = np.argmax(label, axis=1)
label = label.reshape([len(label), 1])

train_confusion_matrix = confusion_matrix(train_y, label)

plt.figure()

plot_confusion_matrix(train_confusion_matrix, classes=[
                      'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], normalize=True)

plt.show()
