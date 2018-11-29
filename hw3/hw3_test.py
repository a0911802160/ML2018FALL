import numpy as np
import pandas as pd
import csv
import math
from keras.models import load_model
from keras import backend as k
import sys


data = pd.read_csv(sys.argv[1], delimiter=',')
x = np.array(data.iloc[:, 1])

test_x = []
for idx in range(len(x)):
    test_x.append(np.reshape(
        np.array(x[idx].split(' '), dtype=np.float64), (1, 48, 48)))
test_x = np.array(test_x)

test_x = test_x/255

model = load_model('best_model.h5')
label = model.predict(test_x)
label = np.argmax(label, axis=1)
label = label.reshape([len(label), 1])

ans = []

for idx in range(len(test_x)):
    ans.append(str(idx)+',')
    ans[idx] += str(label[idx][0])

with open(sys.argv[2], 'w+') as pred_file:
    pred_file.write('id,label\n')
    for idx in range(len(ans)):
        pred_file.write(ans[idx]+'\n')
