import jieba
import sys
import numpy as np
import math
import pandas as pd
import csv
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Bidirectional
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback
from keras.utils import np_utils
from gensim.models import word2vec
import logging
from keras.preprocessing import sequence

import os


def seg_preprocess(test_x_file):
    jieba.load_userdict(sys.argv[2])

    test_X_data = []

    with open(test_x_file, 'r', encoding='utf8') as test_x_file:
        next(test_x_file)
        for line_data in test_x_file:
            now_txt = []
            line_data = line_data.split(',')[1]
            line_data = jieba.cut(line_data, cut_all=False)
            for text in line_data:
                if text != " " and text != "\n":
                    now_txt.append(text)
            test_X_data.append(now_txt)

    test_X_data = np.array(test_X_data)
    np.save('test_X_data.npy', test_X_data)


def generate_embedding_matrix(word2vec_model_path):

    word2vec_model = word2vec.Word2Vec.load(word2vec_model_path)

    embedding_matrix = np.zeros(
        (len(word2vec_model.wv.vocab.items()) + 2, word2vec_model.vector_size))

    word2idx = {'PADDING': 0, 'No_found': 1}

    vocab_list = [(word, word2vec_model.wv[word])
                  for word, _ in word2vec_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 2] = vec
        word2idx[word] = i + 2
    return embedding_matrix, word2idx


def build_RNN_model(embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=False), merge_mode='sum'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=["accuracy"])

    return model


def train_RNN_model(model, train_idx_X, train_y, pretrained=False, pretrained_model=""):

    if pretrained:
        model = load_model(pretrained_model)
        print('load pretrained model: ', pretrained_model)
    np.random.seed(321)
    val_len = int(len(train_idx_X)*.1)
    order = np.arange(len(train_idx_X))
    np.random.shuffle(order)
    val_x = train_idx_X[order[:val_len]]
    val_y = train_y[order[:val_len]]

    train_idx_X = train_idx_X[order[val_len:]]
    train_y = train_y[order[val_len:]]

    checkpoint = ModelCheckpoint('{epoch:05d}-{val_acc:.3f}.h5',
                                 monitor='val_acc', save_best_only=True, period=1)
    model.fit(train_idx_X, train_y,
              batch_size=512,
              epochs=50,
              validation_data=(val_x, val_y),
              callbacks=[checkpoint])

    model.save('RNN.h5')


def generate_X_Y_data(x_data_npy, y_data_csv):
    x_data = np.load(x_data_npy)

    if y_data_csv == "":
        return x_data
    y_data = []
    with open(y_data_csv, 'r') as y_data_csv:
        next(y_data_csv)
        for line_data in y_data_csv:
            y_data.append(line_data.split(',')[1])
    y_data = np.array(y_data, dtype=int)

    return x_data, y_data


def text2idx(X_data, word2idx, max_len):
    idx_data = []
    for line_data in X_data:
        idx_line = []
        for word in line_data:
            if word in word2idx:
                idx_line.append(word2idx[word])
            else:
                idx_line.append(word2idx['No_found'])
        idx_data.append(idx_line)

    idx_data = np.array(idx_data)
    idx_data = sequence.pad_sequences(idx_data, padding='post', maxlen=max_len)
    return idx_data


def predict(model_path, test_x_data, pred_file_path):
    model = load_model(model_path)
    pred = model.predict(test_x_data)
    ans = []

    for idx in range(len(test_x_data)):
        ans.append(str(idx)+',')
        ans[idx] += (str(int(round(pred[idx][0])))+'\n')
        # ans[idx] += (str((pred[idx][0]))[:5]+'\n')

    with open(pred_file_path, 'w+') as pred_file:
        pred_file.write('id,label\n')
        for ans_idx in ans:
            pred_file.write(ans_idx)


if __name__ == '__main__':

    test_x_file, output_file = sys.argv[1], sys.argv[3]
    word2vec_model_path = 'word2vec.model'

    seg_preprocess(test_x_file)
    embedding_matrix, word2idx = generate_embedding_matrix(word2vec_model_path)
    test_x_data = generate_X_Y_data(
        'test_X_data.npy', "")
    test_idx_X = text2idx(test_x_data, word2idx, 128)

    predict('RNN.h5', test_idx_X, output_file)
