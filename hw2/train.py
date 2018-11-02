import numpy as np
import math
import random
import os
import pandas as pd
import sys

train_x_file, train_y_file, test_x_file, output_file = sys.argv[
    1], sys.argv[2], sys.argv[3], sys.argv[4]

# train_set_X/train_set_Y/test_set_X are numpy array
train_set_X = np.array(pd.read_csv(train_x_file, encoding='big5'), dtype=float)
train_set_Y = np.array(pd.read_csv(train_y_file, encoding='big5'), dtype=float)
test_set_X = np.array(pd.read_csv(test_x_file, encoding='big5'), dtype=float)

train_set_X = np.append(train_set_X, np.expand_dims(np.mean(
    train_set_X[..., 11:17], axis=1), -1), axis=1)
train_set_X = np.append(train_set_X, np.expand_dims(np.max(
    train_set_X[..., 11:17], axis=1), -1), axis=1)
train_set_X = np.append(train_set_X, np.expand_dims(np.min(
    train_set_X[..., 11:17], axis=1), -1), axis=1)


test_set_X = np.append(test_set_X, np.expand_dims(np.mean(
    test_set_X[..., 11:17], axis=1), -1), axis=1)
test_set_X = np.append(test_set_X, np.expand_dims(np.max(
    test_set_X[..., 11:17], axis=1), -1), axis=1)
test_set_X = np.append(test_set_X, np.expand_dims(np.min(
    test_set_X[..., 11:17], axis=1), -1), axis=1)


train_set_X = np.append(train_set_X, np.expand_dims(np.mean(
    train_set_X[..., 17:23], axis=1), -1), axis=1)
train_set_X = np.append(train_set_X, np.expand_dims(np.max(
    train_set_X[..., 17:23], axis=1), -1), axis=1)
train_set_X = np.append(train_set_X, np.expand_dims(np.min(
    train_set_X[..., 17:23], axis=1), -1), axis=1)


test_set_X = np.append(test_set_X, np.expand_dims(np.mean(
    test_set_X[..., 17:23], axis=1), -1), axis=1)
test_set_X = np.append(test_set_X, np.expand_dims(np.max(
    test_set_X[..., 17:23], axis=1), -1), axis=1)
test_set_X = np.append(test_set_X, np.expand_dims(np.min(
    test_set_X[..., 17:23], axis=1), -1), axis=1)

data = np.concatenate((train_set_X, test_set_X))

need_normalize_col = [0, 4, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
for i in need_normalize_col:
    mean = np.mean(data[..., i], axis=0)
    std = np.std(data[..., 0], axis=0)
    train_set_X[..., i] = (train_set_X[..., i]-mean)/std
    test_set_X[..., i] = (test_set_X[..., i]-mean)/std

# one-hot-encoding


def one_hot_encoding(train_set_X, class_idx, class_nums, class_shift):
    one_hot_encode = np.zeros((len(train_set_X), class_nums))
    one_hot_encode[np.arange(len(train_set_X)), np.array(
        train_set_X[..., class_idx]-class_shift, dtype=int).transpose()] = 1.0
    return np.append(train_set_X, one_hot_encode, axis=1)


# SEX one-hot-encoding
train_set_X = one_hot_encoding(train_set_X, 1, 2, 1)
test_set_X = one_hot_encoding(test_set_X, 1, 2, 1)

# EDUCATION one-hot-encoding
train_set_X = one_hot_encoding(train_set_X, 2, 7, 0)
test_set_X = one_hot_encoding(test_set_X, 2, 7, 0)

# MARRIAGE one-hot-encoding
train_set_X = one_hot_encoding(train_set_X, 3, 4, 0)
test_set_X = one_hot_encoding(test_set_X, 3, 4, 0)

# PAYMENT one-hot-encoding

for idx in range(5, 11):
    train_set_X = one_hot_encoding(train_set_X, idx, 11, -2)
    test_set_X = one_hot_encoding(test_set_X, idx, 11, -2)

train_set_X = np.delete(train_set_X, (1, 2, 3, 5, 6, 7, 8, 9, 10, 11,
                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), 1)
test_set_X = np.delete(test_set_X, (1, 2, 3, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), 1)


# for idx in range(len(train_set_X)):
#     for i in range(2):
#         train_set_X[idx]=train_set_X[idx].append([0])
#     train_set_X[-train_set_X[idx][1]]
#

# add bias
train_set_X = np.concatenate(
    (np.ones((len(train_set_X), 1)), train_set_X), axis=1)
test_set_X = np.concatenate(
    (np.ones((len(test_set_X), 1)), test_set_X), axis=1)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


weight = np.ones(len(train_set_X[0]))
ada_sum = np.zeros(len(train_set_X[0]))


iteration = 1000
l_rate = 1e-1


for train_iter in range(iteration):

    # test = sigmoid(np.dot(train_set_X, weight)+bias)
    # print(test.shape)
    # test = np.expand_dims(test, -1)
    # print(test.shape)
    # print(test)
    # print(train_set_Y.shape)
    predict = np.array([1 if s >= .5 else 0 for s in sigmoid(
        np.dot(train_set_X, weight)).transpose()])

    error = predict.transpose() - train_set_Y.transpose()
    # print(error)

    grad = np.dot(error, train_set_X)[0]/len(train_set_X)
    # print(grad.shape)
    ada_sum += grad**2
    # print(ada_sum.shape)
    # print(grad)
    # print(weight)
    weight -= l_rate*grad/np.sqrt(ada_sum)
    # print(weight)

    predict = np.array([1 if s >= .5 else 0 for s in sigmoid(
        np.dot(train_set_X, weight)).transpose()])
    # print(len(train_set_X))
    # print(np.sum(np.abs(predict-train_set_Y)))
    accuracy = 1 - \
        np.sum(np.abs(predict-train_set_Y.transpose()))/len(train_set_X)

    print('Iteration: %d   accur: %f' % (train_iter, accuracy))

ans = []

for idx in range(len(test_set_X)):
    ans.append(['id_'+str(idx)])
    predict = [1 if sigmoid(
        np.dot(test_set_X[idx], weight)).transpose() >= 0.5 else 0]
    ans[idx].append(predict[0])

with open(output_file, 'w+') as output:
    output.writelines('id,value\n')
    for i in range(len(ans)):
        output.writelines(str(ans[i][0])+','+str(ans[i][1])+'\n')
