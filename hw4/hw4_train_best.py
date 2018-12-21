import jieba
import sys
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Bidirectional
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LambdaCallback
from keras.utils import np_utils
from gensim.models import word2vec
import logging
from keras.preprocessing import sequence
from keras import backend as K


def train_word2vec_model(train_x_file, test_x_file, dict_file):
    jieba.load_userdict(dict_file)

    trian_X_data = []
    test_X_data = []
    all_X_data = []

    with open(train_x_file, 'r', encoding='utf8') as train_x_file:
        next(train_x_file)
        last_word = ''
        for line_data in train_x_file:
            now_txt = []
            line_data = line_data.split(',')[1]
            line_data = jieba.cut(line_data, cut_all=False)
            for text in line_data:
                if (text[0] == "B" or text[0] == "b")and text[1:].isnumeric():
                    now_txt.append('某人')
                elif text == ' ' and last_word == ' ':
                    continue
                elif text != '\n':
                    now_txt.append(text)
                last_word = text
            trian_X_data.append(now_txt)

    trian_X_data = np.array(trian_X_data)
    np.save('train_X_data.npy', trian_X_data)

    with open(test_x_file, 'r', encoding='utf8') as test_x_file:
        next(test_x_file)
        last_word = ''
        for line_data in test_x_file:
            now_txt = []
            line_data = line_data.split(',')[1]
            line_data = jieba.cut(line_data, cut_all=False)
            for text in line_data:
                if (text[0] == "B" or text[0] == "b")and text[1:].isnumeric():
                    now_txt.append('某人')
                elif text == ' ' and last_word == ' ':
                    continue
                elif text != '\n':
                    now_txt.append(text)
                last_word = text
            test_X_data.append(now_txt)

    test_X_data = np.array(test_X_data)
    np.save('test_X_data.npy', test_X_data)

    all_X_data = np.concatenate((trian_X_data, test_X_data), axis=0)
    model = word2vec.Word2Vec(all_X_data, size=500, iter=30)
    model.save('word2vec.model')


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
                                input_length=100,
                                weights=[embedding_matrix],
                                trainable=False)
    model.add(embedding_layer)
    # model.add(LSTM(64, dropout=0.2, return_sequences=False))
    model.add(Bidirectional(
        LSTM(64, activation='tanh', dropout=0.3, recurrent_dropout=0.3, inner_init='orthogonal', return_sequences=True)))
    model.add(Bidirectional(
        LSTM(32, activation='tanh', dropout=0.2, recurrent_dropout=0.2, inner_init='orthogonal', return_sequences=False)))
    # model.add(LSTM(64, inner_init='orthogonal', return_sequences=False))
    # model.add(Dense(128))
    # model.add(Dropout(0.2))
    # model.add(Dense(64))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-2), metrics=["accuracy"])

    return model


def train_RNN_model(model, train_idx_X, train_y, pretrained=False, pretrained_model=""):

    if pretrained:
        model = load_model(pretrained_model)
        print('#############load pretrained model: ', pretrained_model)
    np.random.seed(5478)
    val_len = int(len(train_idx_X)*.1)
    order = np.arange(len(train_idx_X))
    np.random.shuffle(order)
    val_x = train_idx_X[order[:val_len]]
    val_y = train_y[order[:val_len]]

    train_idx_X = train_idx_X[order[val_len:]]
    train_y = train_y[order[val_len:]]

    checkpoint = ModelCheckpoint('{epoch:05d}-{val_acc:.3f}.h5',
                                 monitor='val_acc', save_weights_only=False, save_best_only=True, period=5, mode='max')
    model.fit(train_idx_X, train_y,
              batch_size=1024,
              validation_data=(val_x, val_y),
              epochs=15,
              initial_epoch=0,
              callbacks=[checkpoint])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_acc', min_delta=0, patience=20, verbose=1)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-2), metrics=["accuracy"])

    model.fit(train_idx_X, train_y,
              batch_size=1024,
              validation_data=(val_x, val_y),
              epochs=250,
              initial_epoch=15,
              callbacks=[checkpoint, reduce_lr, early_stopping])

    model.save('RNN.h5')


def generate_X_Y_data(x_data_npy, y_data_csv):
    x_data = np.load(x_data_npy)

    y_data = []
    with open(y_data_csv, 'r') as y_data_csv:
        next(y_data_csv)
        for line_data in y_data_csv:
            y_data.append(line_data.split(',')[1])
    y_data = np.array(y_data, dtype=int)
    return x_data, y_data


def text2idx(X_data, word2idx, max_len):
    idx_data = []
    # max_string_len = 0
    # sum_of_string_len = 0
    # cnt = 0
    for line_data in X_data:
        # cnt += 1
        # sum_of_string_len += len(line_data)
        # if len(line_data) > max_string_len:
        #     max_string_len = len(line_data)
        #     # print(max_string_len)
        idx_line = []
        for word in line_data:
            if word in word2idx:
                idx_line.append(word2idx[word])
            else:
                idx_line.append(word2idx['No_found'])
        idx_data.append(idx_line)

    idx_data = np.array(idx_data)
    idx_data = sequence.pad_sequences(idx_data, padding='post', maxlen=max_len)
    # print('max string len is', max_string_len)
    # print('avg string len is', sum_of_string_len/cnt)
    return idx_data


if __name__ == '__main__':

    train_x_file, train_y_file, test_x_file, dict_file = sys.argv[
        1], sys.argv[2], sys.argv[3], sys.argv[4]

    word2vec_model_path = 'word2vec.model'

    train_word2vec_model(train_x_file, test_x_file, dict_file)
    embedding_matrix, word2idx = generate_embedding_matrix(word2vec_model_path)
    train_x_data, train_y_data = generate_X_Y_data(
        'train_X_data.npy', train_y_file)
    train_idx_X = text2idx(train_x_data, word2idx, 100)
    model = build_RNN_model(embedding_matrix)
    train_RNN_model(model, train_idx_X, train_y_data,
                    pretrained=False, pretrained_model="RNN.h5")
