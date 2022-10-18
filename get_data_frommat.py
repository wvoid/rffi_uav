import numpy as np
import scipy.io as so
import os


def save_to_traindata():
    # filename = 'F:\matlab\preamble_data\\features_mat_xcorr1'
    # filename1 = 'F:\matlab\preamble_data\\features_mat_xcorr2'
    filename1 = '/home/rs/1/preamble_data/holybro/features_mat_holybro2_524.mat'
    filename2 = '/home/rs/1/preamble_data/holybro/features_mat_holybro1_590.mat'
    filename3 = '/home/rs/1/preamble_data/holybro/features_mat_v51_530.mat'
    filename4 = '/home/rs/1/preamble_data/holybro/features_mat_v52_525.mat'

    data1 = so.loadmat(filename1)
    data2 = so.loadmat(filename2)
    data3 = so.loadmat(filename3)
    data4 = so.loadmat(filename4)
    x1 = data1['features_mat']
    x2 = data2['features_mat']
    x3 = data3['features_mat']
    x4 = data4['features_mat']

    len_x1 = np.size(x1, 0)
    len_x2 = np.size(x2, 0)
    len_x3 = np.size(x3, 0)
    len_x4 = np.size(x4, 0)

    # print(len_x1)
    x = np.vstack((x1, x2, x3, x4))
    y1 = np.zeros((len_x1, 1))
    y2 = np.ones((len_x2, 1))
    y3 = np.empty((len_x3, 1))
    y3[:] = 2
    y4 = np.empty((len_x4, 1))
    y4[:] = 3
    y = np.vstack((y1, y2, y3, y4))
    filename2 = 'data_train/data_4rfs'
    np.savez(filename2, data=x, label=y)


def get_mat(filename):
    loacation = os.path.join('/home/rs/1/preamble_data', filename)
    # loacation = os.path.join('F:\matlab\preamble_data', filename)
    data = so.loadmat(loacation)
    return data[filename]
