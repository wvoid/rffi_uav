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
from get_data_frommat import save_to_traindata, get_mat, plt_cm

if __name__ == '__main__':
    # save_to_traindata()
    filename = 'data_train/data_4rfs_1.npz'
    data = np.load(filename)
    data_pb_ps = data['data'].real
    label = data['label']

    # pca = PCA(n_components=5)
    # data_pb_ps = pca.fit_transform(data_pb_ps)
    # data_pb_ps_x1, data_pb_ps_x2 = np.empty((1, 2)), np.empty((1, 2))
    # for i in range(1200):
    #     if label[i] == 0:
    #         data_pb_ps_x1 = np.vstack((data_pb_ps_x1, data_pb_ps[i]))
    #     else:
    #         data_pb_ps_x2 = np.vstack((data_pb_ps_x2, data_pb_ps[i]))
    # x1 = [row[0] for row in data_pb_ps_x1]
    # y1 = [row[1] for row in data_pb_ps_x1]
    # x2 = [row[0] for row in data_pb_ps_x2]
    # y2 = [row[1] for row in data_pb_ps_x2]
    # # print(y)
    # plt.scatter(x1, y1, c='r', s=3)
    # plt.scatter(x2, y2, c='b', s=3)
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(data_pb_ps, label, test_size=0.3)

    randomforest = RandomForestClassifier(random_state=42, n_estimators=100)
    # svm_clf=svm.SVC()

    randomforest.fit(x_train, y_train)
    # svm_clf.fit(x_train,y_train.ravel())

    y_pred = randomforest.predict(x_test)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(randomforest, x_test, y_test)
    # plot_confusion_matrix(svm_clf,x_test,y_test)
    # c = confusion_matrix(y_test, y_pred)
    # print(c)
    plt.show()
    x_pb_test = get_mat('features_mat_v52_197')  # .real
    # x_pb_test = pca.transform(x_pb_test)
    print(randomforest.predict(x_pb_test))
    # print(svm_clf.predict(x_pb_test))
    # plt_cm(c)

