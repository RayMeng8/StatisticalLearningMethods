# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn.svm import SVC

class SVM:
    #gaussian kernel function/linear kernel function
    def __init__(self, max_iter = 3000, kernel="linear"):
        self.max_iter = max_iter
        self.kernel = kernel

    def init_args(self, features, labels):
        self.num_samples = features.shape[0]
        self.num_features = features.shape[1]
        self.train_X = features
        self.train_y = labels

        self.alpha = np.ones(self.num_samples)
        self.b = 0.0
        self.C = 10

        self.E = np.zeros(self.num_samples)
        for i in range(self.num_samples):

            self.E[i] = self.calc_E(i)

    #计算三种核函数
    def calc_kernel(self, x_j, x_i):
        if self.kernel == "linear":
            return sum([x_j[k] * x_i[k] for k in range(self.num_features)])
        if self.kernel == "poly":
            return (sum([x_j[k] * x_i[k] for k in range(self.num_features)])+1)**2

    #若当前alpha为最优解，则充分必要条件为KTT条件，可写为这样
    #每次选择违反KKT条件最严重的点
    def is_KKT_satisfied(self, i):
        yg_i = self.train_y[i] * self.calc_g(i)
        print("index : " + str(i))
        print("alpha[i] : " + str(self.alpha[i]))
        print("y_i * g_i : " + str(yg_i))
        if self.alpha[i] == 0 :
            return yg_i>=1
        elif self.alpha[i] < self.C and self.alpha[i]>0:
            return yg_i==1
        else:
            return yg_i<=1

    #选择两个alpha
    def select_two_alpha(self):
        index_list = [i for i in range(self.num_samples) if i < self.C and i >0]
        #先遍历间隔边界上的，再遍历整个数据集上的
        oppose_index_list = [i for i in range(self.num_samples) if i not in index_list]
        index_list.extend(oppose_index_list)
        for i in index_list:
            if self.is_KKT_satisfied(i):
                continue
            if self.E[i]>0:
                j = min(range(self.num_samples), key=lambda x:self.E[x])
            else:
                j = max(range(self.num_samples), key=lambda x:self.E[x])
            return i, j
        return -1, -1

    #计算差值
    def calc_E(self, i):
        return self.calc_g(i)-self.train_y[i]

    def calc_g(self, i):
        sum = self.b
        for k in range(self.num_samples):
            sum += self.alpha[k] * self.train_y[k] * self.calc_kernel(self.train_X[k], self.train_X[i])
        return sum

    #添加停机条件
    def fit(self, features, labels):
        self.init_args(features, labels)
        for it in range(self.max_iter):
            i, j=self.select_two_alpha()
            if self.train_y[i] == self.train_y[j]:
                L = max(0, self.alpha[i]+self.alpha[j]-self.C)
                H = min(self.C, self.alpha[i]+self.alpha[j])
            else:
                L = max(0, self.alpha[j]-self.alpha[i])
                H = min(self.C, self.C+self.alpha[j]-self.alpha[i])
            eta = self.calc_kernel(self.train_X[i], self.train_X[i]) + self.calc_kernel(self.train_X[j], self.train_X[j]) - 2 * self.calc_kernel(self.train_X[i], self.train_X[j])
            E_i = self.E[i]
            E_j = self.E[j]
            alpha_i = self.alpha[i]
            alpha_j = self.alpha[j]
            b = self.b
            y_i = self.train_y[i]
            y_j = self.train_y[j]
            alpha_j_new_unc = self.alpha[j] + self.train_y[j] * (self.E[i] - self.E[j])/eta
            alpha_j_new = alpha_j_new_unc
            if alpha_j_new<L:
                alpha_j_new = L
            elif alpha_j_new>H:
                alpha_j_new = H
            alpha_i_new = self.alpha[i] + self.train_y[i] * self.train_y[j] *(self.alpha[j] - alpha_j_new)
            b_i_new = -self.E[i] - self.train_y[i] * self.calc_kernel(self.train_X[i], self.train_X[i]) * (alpha_i_new - self.alpha[i]) - self.train_y[j] * self.calc_kernel(self.train_X[j], self.train_X[i]) * (alpha_j_new - self.alpha[j]) + self.b
            b_j_new = -self.E[j] - self.train_y[i] * self.calc_kernel(self.train_X[i], self.train_X[j]) * (alpha_i_new - self.alpha[i]) - self.train_y[j] * self.calc_kernel(self.train_X[j], self.train_X[j]) * (alpha_j_new - self.alpha[j]) + self.b
            if alpha_i_new >0 and alpha_i_new < self.C:
                b_new = b_i_new
            elif alpha_j_new>0 and alpha_j_new<self.C:
                b_new = b_j_new
            else:
                b_new = (b_i_new + b_j_new)/2.0
            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new
            self.b = b_new

            self.E[i] = self.calc_E(i)
            self.E[j] = self.calc_E(j)
        print("train_done!")

    #二分类
    def predict(self, X):
        sum = self.b
        for k in range(self.num_samples):
            sum += self.alpha[k] * self.train_y[k] * self.calc_kernel(self.train_X[k], X)
        if sum>0:
            return 1
        else:
            return -1

    def score(self, X_test, y_test):
        right_cnt = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                right_cnt += 1
        return right_cnt/len(X_test)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = df.iloc[0:100, :]
data = np.array(data)
X, y = data[:, 0:-1], data[:, -1]
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = SVM()
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print(score)

sklearn_svm = SVC()
sklearn_svm.fit(X_train, y_train)
score = sklearn_svm.score(X_test, y_test)
print(score)
