import numpy as np
import sys

weight = np.load('weight.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')


data = []

for _ in range(18):
    data.append([])

with open(sys.argv[1], 'r', encoding='big5') as test_data:
    row_idx = 0
    tmp = []
    for row in test_data:
        row = row.split(',')
        # 10 hr data
        for col_idx in range(2, 11):
            if row[col_idx].startswith('NR'):
                data[row_idx % 18] = np.append(
                    data[row_idx % 18], float(.0))
            else:
                data[row_idx % 18] = np.append(
                    data[row_idx % 18], row[col_idx])
        row_idx += 1
data = np.array(data)

data = np.delete(data, (1, 3, 4, 6, 8, 10, 13, 14,  17), 0)

WIND_DIR = np.array(data[7], dtype=float)
WIND_DIR = WIND_DIR*np.pi/360.0
WIND_SPD = np.array(data[8], dtype=float)

data = np.append(data, [np.cos(WIND_DIR)*WIND_SPD], axis=0)
data = np.append(data, [np.sin(WIND_DIR)*WIND_SPD], axis=0)
data = np.delete(data, 7, 0)
test_set = []

for row_idx in range(0, len(data[0]), 9):
    tmp = []
    for i in range(9):
        tmp2 = ([float(i) if i != '.' else 0.0 for i in(data[:, row_idx+i])])
        tmp = np.concatenate((tmp, tmp2))

    test_set.append(np.array(tmp, dtype=float))

# feature scaling with the same scale as train_set
test_set = (test_set-mean)/std

# add bias
test_set = np.concatenate(
    (np.ones((len(test_set), 1)), test_set), axis=1)

ans = []

for idx in range(len(test_set)):
    ans.append(['id_'+str(idx)])
    predict = np.dot(weight, test_set[idx])
    ans[idx].append(predict)

with open(sys.argv[2], 'w+', encoding='big5') as pred_file:
    pred_file.writelines('id,value\n')
    for pred in ans:
        pred_file.writelines(str(pred[0])+','+str(pred[1])+'\n')
