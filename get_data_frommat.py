import numpy as np
import scipy.io as so
import os


def save_to_traindata():
    filename = 'F:\matlab\preamble_data\\pb_mat_600_rf1'
    filename1 = 'F:\matlab\preamble_data\\pb_mat_600_rf2'
    # filename = '/home/rs/1/preamble_data/features_mat1.mat'
    # filename1 = '/home/rs/1/preamble_data/features_mat2.mat'
    data1 = so.loadmat(filename)
    data2 = so.loadmat(filename1)
    x1 = data1['pb_mat']
    x2 = data2['pb_mat']
    x = np.vstack((x1, x2))
    y1 = np.zeros((600, 1))
    y2 = np.ones((600, 1))
    y = np.vstack((y1, y2))
    filename2 = 'data_train/data_rf1_rf2_test'
    np.savez(filename2, data=x, label=y)


def get_mat(filename):
    #loacation = os.path.join('/home/rs/1/preamble_data', filename)
    loacation = os.path.join('F:\matlab\preamble_data', filename)
    data = so.loadmat(loacation)
    return data[filename]
