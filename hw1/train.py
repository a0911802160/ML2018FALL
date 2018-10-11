import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
import datetime

NOW_WEIGHT = ''


def generate_model_weight():

    data = []

    for _ in range(19):
        data.append([])

    with open('train.csv', 'r', encoding='big5') as csv_file:
        next(csv_file)  # skip header
        row_idx = 0
        for row_data in csv_file:
            row_data = row_data.split(',')
            if row_idx % 18 == 0:
                for _ in range(24):
                    data[0] = np.append(data[0], row_data[0])
            for i in range(24):
                if row_data[i+3].startswith('NR'):

                    data[(row_idx % 18) +
                         1] = np.append(data[(row_idx % 18)+1], (float(.0)))
                else:
                    data[(row_idx % 18)+1] = np.append(data[(row_idx % 18)+1],
                                                       (float(row_data[i+3])))
            row_idx += 1

    data = np.array(data)

    data = np.delete(data, (2, 4, 5, 7, 9, 11, 14, 15, 18), 0)
    # print(data[8])
    WIND_DIR = np.array(data[8], dtype=float)
    WIND_DIR = WIND_DIR*np.pi/360.0
    WIND_SPD = np.array(data[9], dtype=float)

    data = np.append(data, [np.cos(WIND_DIR)*WIND_SPD], axis=0)
    data = np.append(data, [np.sin(WIND_DIR)*WIND_SPD], axis=0)
    # print(data[16])
    data = np.delete(data, 8, 0)
    # os.system('pause')

    # data[]
    # for i in range(1, len(data)):
    #     tmp = np.array(data[i], dtype=float)
    #     plt.scatter(data[9], tmp**2,
    #                 s=np.pi*3, c=(0, 0, 0), alpha=0.2)
    #     plt.xlabel(i)
    #     plt.savefig('feature'+str(i)+'.png')
    #     plt.gcf().clear()

    # with open('now_data.txt', 'w+') as f:
    #     for row in range(len(data[0])-1):
    #         if row < 60:
    #             print(data[:, row].tolist())
    #             f.writelines(str(data[:, row].tolist()))
    #             f.writelines('\n')

    # train_set=[[train_x],[train_y]]
    train_set = [[], []]

    col_mean = []
    for i in range(len(data)-1):
        col_mean.append(np.mean(np.array(data[i+1, :], dtype=float)))

    # print(data[5])
    # os.system('pause')
    for row_idx in range(len(data[0])-9):
        # check if there is continuous 10 hrs(will fail in the end of a month)
        is_continuous = True
        tmp_date = data[0, row_idx].split('/')[1]

        for i in range(10):
            # or data[1, row_idx+i] == '0.0':
            if tmp_date != data[0, row_idx+i].split('/')[1]:
                is_continuous = False
                break
        if is_continuous:
            # use first 9 hr data as train set_x
            tmp = []
            for i in range(9):
                if data[1, row_idx+i] == '0.0':
                    tmp = np.concatenate((tmp, col_mean))
                else:
                    tmp2 = (
                        [float(i) if i != '.' else 0.0 for i in(data[1:, row_idx+i])])
                    tmp = np.concatenate((tmp, tmp2))
            train_set[0].append(np.array(tmp, dtype=float))
            # use the 10th hr PM2.5 as train set_y
            train_set[1].append(float(data[5, row_idx+9]))

    train_set[0] = np.array(train_set[0])
    print(len(train_set[0]))
    os.system("pause")

    # feature scaling before adding bias(bias will force div0 problem in f.s)
    mean = []
    std = []
    for i in range(len(train_set[0][0])):
        mean.append(np.min(train_set[0][:, i]))
        std.append(
            np.max(train_set[0][:, i])-np.min(train_set[0][:, i]))

    np.save('mean.npy', mean)
    np.save('std.npy', std)

    train_set[0] = (train_set[0]-mean)/std

    # add x^2
    # train_set[0] = np.concatenate(
    #     (train_set[0], np.power(train_set[0], 2)), axis=1)

    # add bias
    train_set[0] = np.concatenate(
        (np.ones((len(train_set[0]), 1)), train_set[0]), axis=1)

    # add weight
    # weight = np.ones(len(train_set[0][0]))

    # train_epoch = 100000
    train_epoch = 60000

    def shuffle_in_unison_scary(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    # pitch = 12
    # patch = int(len(train_set[0])/pitch)

    # best_i = 0
    # best_rmse = 80000

    # for i in range(pitch):
    #     weight = np.ones(len(train_set[0][0]))
    #     learning_rate = .001
    #     grad_learning_rate = 11111.1
    #     bias = 0.0
    #     bias_sum_grad = 0.0
    #     sum_grad_step2 = np.zeros(len(train_set[0][0]))

    #     ori = train_set.copy()
    #     now_train_set = []
    #     tmp = np.delete(ori[0], range(i*patch, (i+1)*patch), 0)
    #     # now_train_set[0] = np.concatenate(now_train_set[0], tmp)
    #     now_train_set .append(np.delete(
    #         ori[0], range(i*patch, (i+1)*patch), 0))
    #     now_train_set .append(np.delete(
    #         ori[1], range(i*patch, (i+1)*patch), 0))
    #     for train_iter in range(train_epoch):
    #         # order = np.array(range(len(train_set[0])))
    #         # np.random.shuffle(order)
    #         # has_updated = False
    #         y_pred = np.dot(now_train_set[0], weight)+bias  # +bias
    #         train_error = now_train_set[1]-y_pred
    #         RMSE = math.sqrt(
    #             np.sum(np.power(train_error, 2))/len(now_train_set[0]))
    #         grad = -2*np.dot(now_train_set[0].transpose(), train_error
    #                          )/len(now_train_set[0]) + 2*.01*weight/len(now_train_set[0])
    #         bias_grad = -2*np.sum(train_error)/len(now_train_set[0])
    #         # bias_grad = -2.0*np.sum(train_error)
    #         sum_grad_step2 += grad**2
    #         bias_sum_grad += bias_grad**2
    #         # sum_bias_grad += bias_grad**2
    #         weight -= grad_learning_rate*grad/np.sqrt(sum_grad_step2)
    #         bias -= grad_learning_rate*bias_grad/np.sqrt(bias_sum_grad)
    #         # bias -= learning_rate*bias_grad/np.sqrt(sum_bias_grad)
    #         # print('train iteration: %d ,RMSE= %f' % (train_iter, RMSE))

    #     y_pred = np.dot(train_set[0], weight)+bias  # +bias
    #     train_error = train_set[1]-y_pred
    #     RMSE = math.sqrt(
    #         np.sum(np.power(train_error, 2))/len(train_set[0]))
    #     print('validation: %d ,RMSE= %f' % (i, RMSE))

    weight = np.ones(len(train_set[0][0]))
    # for i in range(9):
    #     weight[1+9*i:1+9*(i+1)] *= 0.8**(8-i)
    # print('weight :', weight)
    # os.system('pause')
    bias = 0.0
    bias_sum_grad = 0.0
    sum_grad_step2 = np.zeros(len(train_set[0][0]))

    # ori = train_set.copy()
    # now_train_set = []
    # # now_train_set[0] = np.concatenate(now_train_set[0], tmp)
    # now_train_set .append(np.delete(
    #     ori[0], range(i*patch, (i+1)*patch), 0))
    # now_train_set .append(np.delete(
    #     ori[1], range(i*patch, (i+1)*patch), 0))

    alpha = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    m_t = 0
    v_t = 0
    RMSEchart = []
    for train_iter in range(1, train_epoch):
        y_pred = np.dot(train_set[0], weight)   # +bias
        train_error = train_set[1]-y_pred
        # RMSE = math.sqrt(
        #     np.sum(np.power(train_error, 2))/len(now_train_set[0]))
        grad = -2*np.dot(train_set[0].transpose(), train_error.transpose()
                         )/len(train_set[0]) + 2*.1*weight/len(train_set[0])
        # bias_grad = -2*np.sum(train_error)/len(train_set[0])
        RMSE = math.sqrt(
            np.sum(np.power(train_error, 2))/len(train_set[0]))
        print('validation: %d ,RMSE= %f' % (train_iter, RMSE))
        RMSEchart.append(RMSE)
        # bias_grad = -2.0*np.sum(train_error)
        # sum_grad_step2 += grad**2
        # bias_sum_grad += bias_grad**2
        # sum_bias_grad += bias_grad**2
        m_t = beta_1*m_t + (1-beta_1)*grad
        v_t = beta_2*v_t + (1-beta_2)*(grad*grad)
        m_cap = m_t/(1-(beta_1**train_iter))
        v_cap = v_t/(1-(beta_2**train_iter))
        weight_prev = weight.copy()
        weight -= (alpha*m_cap)/(np.sqrt(v_cap) +
                                 epsilon)
        # bias -= alpha*bias_grad/np.sqrt(bias_sum_grad)
        # print(bias)
        if(weight.all == weight_prev.all):
            break
    plt.plot(np.arange(0, len(RMSEchart), 1), RMSEchart, label='ADAM')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('RMSE_chart.png')
    plt.show()

    NOW_WEIGHT = 'ada_gradient_weight_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")+'.npy'
    print('save weight as', NOW_WEIGHT)
    np.save(NOW_WEIGHT, weight)
    np.save('bias.npy', bias)
    NOW_WEIGHT = 'weight.npy'
    np.save(NOW_WEIGHT, weight)
    return NOW_WEIGHT


def predict_with_weight(use_weight):

    weight = np.load('weight.npy')
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    bias = np.load('bias.npy')

    data = []

    for _ in range(18):
        data.append([])

    with open('test.csv', 'r', encoding='big5') as test_data:
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

    # data = np.delete(data, (0, 1, 10, 11, 13, 16, 17), 0)
    data = np.delete(data, (1, 3, 4, 6, 8, 10, 13, 14,  17), 0)
    # print(data[15])

    WIND_DIR = np.array(data[7], dtype=float)
    WIND_DIR = WIND_DIR*np.pi/360.0
    WIND_SPD = np.array(data[8], dtype=float)

    data = np.append(data, [np.cos(WIND_DIR)*WIND_SPD], axis=0)
    data = np.append(data, [np.sin(WIND_DIR)*WIND_SPD], axis=0)
    # print(data[15])
    data = np.delete(data, 7, 0)
    # print(len(data))
    # os.system('pause')
    test_set = []

    for row_idx in range(0, len(data[0]), 9):
        tmp = []
        for i in range(9):
            tmp2 = ([float(i) if i != '.' else 0.0 for i in(data[:, row_idx+i])])
            tmp = np.concatenate((tmp, tmp2))

        test_set.append(np.array(tmp, dtype=float))
        # train_set[1].append(float(data[10, row_idx+9]))

    # feature scaling with the same scale as train_set
    test_set = (test_set-mean)/std

    # add x^2 and x^3
    # test_set = np.concatenate(
    #     (test_set, np.power(test_set, 2)), axis=1)

    # add bias
    test_set = np.concatenate(
        (np.ones((len(test_set), 1)), test_set), axis=1)

    ans = []

    for idx in range(len(test_set)):
        ans.append(['id_'+str(idx)])
        predict = np.dot(weight, test_set[idx])  # +bias
        ans[idx].append(predict)

    with open('predict' +
              datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")+'.csv', 'w+', encoding='big5') as pred_file, open('now_predict.csv', 'w+', encoding='big5') as now_file:
        pred_file.writelines('id,value\n')
        now_file.writelines('id,value\n')

        for pred in ans:
            pred_file.writelines(str(pred[0])+','+str(pred[1])+'\n')
            now_file.writelines(str(pred[0])+','+str(pred[1])+'\n')
    print('save now_predict.csv')


if __name__ == '__main__':
    NOW_WEIGHT = generate_model_weight()
    predict_with_weight(NOW_WEIGHT)
