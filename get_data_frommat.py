import numpy as np
import scipy.io as so
import os
from matplotlib import pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.4)

num = 1
label_onehot = np.eye(num).astype(np.int_)
label=[0,1,2,3]


def save_to_traindata():
    filename = []
    # filename = 'F:\matlab\preamble_data\\features_mat_xcorr1'
    # filename1 = 'F:\matlab\preamble_data\\features_mat_xcorr2'
    filename.append('/home/rs/1/preamble_data/holybro/features_mat_holybro2_524.mat')
    filename.append('/home/rs/1/preamble_data/holybro/features_mat_holybro1_590.mat')
    filename.append('/home/rs/1/preamble_data/holybro/features_mat_v51_530.mat')
    filename.append('/home/rs/1/preamble_data/holybro/features_mat_v52_525.mat')

    len = 0
    x=[]
    y=[]
    for i in range(np.size(filename)):
        data = so.loadmat(filename[i])
        x.append(data['features_mat'])
        len=np.size(data['features_mat'],0)
        print(len)
        y_temp=np.empty((len,num))
        y_temp[:]=label[i]
        y.append(y_temp)
    x=np.array(x)
    y=np.array(y)
    x=np.vstack((x[:]))
    y=np.vstack((y[:])).astype(np.int_)
    # print(x[523:526],y[523:526])
    # data1 = so.loadmat(filename1)
    # data2 = so.loadmat(filename2)
    # data3 = so.loadmat(filename3)
    # data4 = so.loadmat(filename4)
    # x1 = data1['features_mat']
    # x2 = data2['features_mat']
    # x3 = data3['features_mat']
    # x4 = data4['features_mat']
    #
    # len_x1 = np.size(x1, 0)
    # len_x2 = np.size(x2, 0)
    # len_x3 = np.size(x3, 0)
    # len_x4 = np.size(x4, 0)
    #
    # # print(len_x1)
    # x = np.vstack((x1, x2, x3, x4))
    # y1 = np.zeros((len_x1, 1))
    # y2 = np.ones((len_x2, 1))
    # y3 = np.empty((len_x3, 1))
    # y3[:] = 2
    # y4 = np.empty((len_x4, 1))
    # y4[:] = 3
    # y = np.vstack((y1, y2, y3, y4))
    filename2 = 'data_train/data_4rfs_1'
    np.savez(filename2, data=x, label=y)


def get_mat(filename):
    loacation = os.path.join('/home/rs/1/preamble_data/test/', filename)
    # loacation = os.path.join('F:\matlab\preamble_data', filename)
    data = so.loadmat(loacation)
    return data['features_mat']
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

