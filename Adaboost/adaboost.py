# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class Adaboost:
    def __init__(self, num_g = 100, learning_r = 1.0):
        self.num_g = num_g
        self.learning_r = learning_r

    def init_args(self, features, labels):
        self.train_X = features
        self.train_y = labels
        self.num_samples = features.shape[0]
        self.num_features = features.shape[1]

        self.w = [1.0/self.num_samples] * self.num_samples
        self.alpha = np.array([0.0] * self.num_g)
        self.g_set = []

    #计算对于第i个feature最好的间隔点，分类结果，方向
    def calc_g_feature(self, i):
        error = self.num_samples
        X = self.train_X[:, i]
        y = self.train_y
        min_datai = min(X)
        max_datai = max(X)
        step = (max_datai-min_datai)//self.learning_r

        res_point = 0.0
        direct = None
        res_arr = np.array([0.0] * self.num_samples)

        for j in range(1, int(step)):
            point = min_datai+self.learning_r*j
            if point not in X:
                res_arr_pos = np.array([1 if X[k]>point else -1 for k in range(self.num_samples)])
                err_pos = sum([self.w[k] for k in range(self.num_samples) if res_arr_pos[k] != y[k] ])
                res_arr_neg = np.array([1 if X[k]<point else -1 for k in range(self.num_samples)])
                err_neg = sum([self.w[k] for k in range(self.num_samples) if res_arr_neg[k] != y[k] ])

                if err_pos<err_neg:
                    if err_pos<error:
                        res_point = point
                        direct = "pos"
                        res_arr = res_arr_pos
                        error = err_pos
                else:
                    if err_neg<error:
                        res_point = point
                        direct = "neg"
                        res_arr = res_arr_neg
                        error = err_neg
        return res_point, direct, res_arr, error

    def fit(self, features, labels):
        self.init_args(features, labels)

        for iter in range(self.num_g):
            res_error = self.num_samples
            res_point = 0.0
            res_direct = None
            res_arr = np.array([0.0] * self.num_samples)
            res_dimension = -1
            for dimension in range(self.num_features):
                point, direct, arr, error = self.calc_g_feature(dimension)
                if error<res_error:
                    res_point = point
                    res_direct = direct
                    res_dimension = dimension
                    res_arr = arr
                    res_error = error
                if res_error == 0:
                    break
            #calc alpha
            alpha = 1 / 2 * np.log((1-res_error)/res_error)
            self.alpha[iter] = alpha
            #renew g_set
            self.g_set.append((res_dimension, res_point, res_direct, res_arr))
            #calc Z
            Z = sum([self.w[k] * np.exp(-1* alpha * self.train_y[k] * res_arr[k]) for k in range(self.num_samples)])
            #renew w
            for k in range(self.num_samples):
                self.w[k] = self.w[k] * np.exp(-1* alpha * self.train_y[k] * res_arr[k]) / Z

    def predict(self, x):
        res = 0.0
        for i in range(self.num_g):
            #怎么能直接用arr？？？
            dimension, point, direct, train_arr = self.g_set[i]
            if direct=="pos":
                if x[dimension] > point:
                    res_i = 1
                else:
                    res_i = -1
            else:
                if x[dimension]>point:
                    res_i = -1
                else:
                    res_i = 1
            res += self.alpha[i] * res_i
        print(res)
        if res>0:
            return 1
        return -1

    def score(self, testX, testy):
        right_cnt = 0
        for i in range(len(testX)):
            x = testX[i]
            if self.predict(x)==testy[i]:
                right_cnt+=1
        return right_cnt/len(testX)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = df.iloc[0:100, [0, 1, -1]]
data = np.array(data)
X, y = data[:, 0:-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

adb = Adaboost(num_g=10, learning_r=0.2)
adb.fit(X_train, y_train)
score = adb.score(X_test, y_test)
print("Test score is", score)