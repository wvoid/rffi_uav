import numpy as np
import scipy.io as so
import os
from matplotlib import pyplot as plt
import seaborn as sns
import fd

# sns.set(font_scale=1.4)

num = 6
label_onehot = np.eye(num).astype(np.int_)
label = [0, 1, 2, 3, 4, 5]


def get_fd(data, label):
    length = len(data)
    t = np.zeros((length, 32))
    for i in range(32):
        print(i)
        t[:, i] = [fd.hfd(row) for row in data[:, i * 50:(i + 1) * 50]]
    filename2 = 'data_train/data_6rf_fd_decfo_11.9与11.8同型号'

    np.savez(filename2, data=t, label=label)


def save_to_traindata():
    x = []
    y = []
    for i in range(6):
        for j in ['1']:
            filename = f'F:\matlab\preamble_data\\11.9_与11.8相同usrp型号\\deCFO\\pb_mat_{i + 1}_deCFO'
            # filename = f'F:\matlab\preamble_data\\11.9\\snr\\pb_mat_{i + 1}'
            data = so.loadmat(filename)
            x.append(data['pb_mat'])
            len = np.size(data['pb_mat'], 0)
            print(len)
            y_temp = np.empty((len, num))
            y_temp[:] = label_onehot[i]
            y.append(y_temp)
    x = np.array(x)
    y = np.array(y)
    x = np.vstack((x[:]))
    y = np.vstack((y[:])).astype(np.int_)
    filename2 = 'data_train/data_6rf_rawiq_onehot_nosnr_decfo_11.9_与11.8相同usrp型号'
    np.savez(filename2, data=x, label=y)


def get_mat(filename):
    loacation = os.path.join('F:\matlab\preamble_data\\11.8', filename)
    # loacation = os.path.join('F:\matlab\preamble_data', filename)
    data = so.loadmat(loacation)
    return data['pb_mat']


def plt_cm(c):
    f, ax = plt.subplots(figsize=(8, 6))
    df_cm = c
    sns.heatmap(df_cm, annot=True, vmax=100.0, vmin=0.0, fmt='.1f', cmap='Greys', annot_kws={'size': 14})
    label_x = ax.get_xticklabels()
    label_y = ax.get_yticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right', fontsize=16)
    plt.setp(label_y, fontsize=16)
    plt.xlabel('Predicted Emotion', fontsize=20)
    plt.ylabel('True Emotion', fontsize=20)
    # plt.savefig('./fig_confusion_matrix_of_label_embedding_prediction.png')
    plt.show()
