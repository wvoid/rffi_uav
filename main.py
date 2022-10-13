import os.path

import numpy as np
import scipy.io as so
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
from get_data_frommat import save_to_traindata, get_mat

if __name__ == '__main__':
    #save_to_traindata()
    filename = 'data_train/data_rf1_rf2_ps_test.npz'
    data = np.load(filename)
    data_pb_ps = data['data'].real
    label = data['label']
    # plt.plot(data_pb_ps[599])
    # plt.show()
    target_name = [0, 1]
    x_train, x_test, y_train, y_test = train_test_split(data_pb_ps, label, test_size=0.3)
    randomforest = RandomForestClassifier(random_state=42, n_estimators=100)
    print(x_train.shape,y_train.shape)
    randomforest.fit(x_train, y_train.ravel())
    y_pred = randomforest.predict(x_test)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(randomforest, x_test, y_test)
    plt.show()
    x_pb_test = get_mat('features_mat_test2')
    print(randomforest.predict(x_pb_test))
