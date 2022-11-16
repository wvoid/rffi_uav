import os.path
import numpy as np
import scipy.io as so

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from get_data_frommat import save_to_traindata, get_mat,get_fd, plt_cm
from dnn_rffi import complex2iq
import fd





if __name__ == '__main__':
    # save_to_traindata()
    filename = 'data_train/data_6rf_rawiq_11.8.npz'
    data = np.load(filename)
    data_pb_ps = data['data']
    label = data['label']
    # print(label[4000-10:4000+10])
    # print(data_pb_ps[1,:])
    # get_fd(data_pb_ps,label)

    filename1 = 'data_train/data_6rf_fd_decfo_11.9与11.8同型号.npz'
    data1 = np.load(filename1)
    data_pb_ps1 = data1['data']
    label1 = data1['label']

    # print(label[4000-10:4000+10])
    data_pb=data_pb_ps[1]
    fd_pb=np.zeros((1,32))
    for i in range(32):
        fd_pb[0,i]=fd.hfd(data_pb[ i * 50:(i + 1) * 50])
    print(fd_pb)
    # d=data_pb_fd[1,50:100]
    # print(fd.hfd(d))
    plt.plot(data_pb.real)
    plt.show()

    # # # pca = PCA(n_components=2)
    # # # data_pb_ps = pca.fit_transform(data_pb_ps)
    # # # data_pb_ps_x1, data_pb_ps_x2 = np.empty((1, 2)), np.empty((1, 2))
    # # # for i in range(1200):
    # # #     if label[i] == 0:
    # # #         data_pb_ps_x1 = np.vstack((data_pb_ps_x1, data_pb_ps[i]))
    # # #     else:
    # # #         data_pb_ps_x2 = np.vstack((data_pb_ps_x2, data_pb_ps[i]))
    # # # x1 = [row[0] for row in data_pb_ps_x1]
    # # # y1 = [row[1] for row in data_pb_ps_x1]
    # # # x2 = [row[0] for row in data_pb_ps_x2]
    # # # y2 = [row[1] for row in data_pb_ps_x2]
    # # # # print(y)
    # # # plt.scatter(x1, y1, c='r', s=3)
    # # # plt.scatter(x2, y2, c='b', s=3)
    # # # plt.show()
    # #
    # x_train, x_test, y_train, y_test = train_test_split(data_pb_ps, label, test_size=0.1,random_state=1)
    # x_train1, x_test1, y_train1, y_test1 = train_test_split(data_pb_ps1, label1, test_size=0.9,random_state=1)
    #
    # randomforest = RandomForestClassifier(random_state=42, n_estimators=50)
    # # svm_clf=svm.SVC()
    #
    # randomforest.fit(x_train, y_train.ravel())
    # # svm_clf.fit(x_train,y_train.ravel())
    #
    # y_pred1 = randomforest.predict(x_test1)
    # # print(y_pred.shape,y_test.shape)
    # print(classification_report(y_test1, y_pred1))
    # plot_confusion_matrix(randomforest, x_test1, y_test1)
    # # plot_confusion_matrix(svm_clf,x_test,y_test)
    # # c = confusion_matrix(y_test, y_pred)
    # # print(c)
    # plt.show()
    # # x_pb_test = get_mat('pb_mat_v2').real
    # # x_pb_test = pca.transform(x_pb_test)
    # # print(randomforest.predict(x_pb_test))
    # # print(svm_clf.predict(x_pb_test))
    # # plt_cm(c)

