# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LR:
    def __init__(self, max_iter=200, learning_rate=0.1, p=0.5):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.p = p

    def fit(self, X, y):
        self.weights = np.array([0.0] * X[0].size, dtype=float)
        for i in range(self.max_iter):
            grads = np.array([0.0] * X[0].size, dtype=float)
            for m, n in zip(X, y):
                # avoid overflow encountered in exp
                inx = np.dot(m, self.weights)
                if inx >= 0:
                    h = n - 1.0/(1+np.exp(-inx))
                else:
                    h = n - np.exp(inx)/(1+np.exp(inx))
                #h = n - 1.0/(1+np.exp(-np.dot(m, self.weights)))
                for j in range(X[0].size):
                    #似然函数最大，梯度上升法
                    #似然函数对w求偏导得
                    #grad = sum(x_i * (y_i - sigmoid(w * x_i)))
                    grads[j] += m[j] * h
            self.weights += grads
        return self.weights

    def score(self, X, y):
        right_cnt = 0.0
        self.odds = np.log(self.p/(1-self.p))
        for i, j in zip(X, y):
            result = np.dot(i.T, self.weights)
            if (result>self.odds and j==1) or (result<self.odds and j==0):
                right_cnt += 1
        return right_cnt / len(X)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
df['label'] = iris.target
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, 0:-1], data[:, -1]
tmp = np.ones((X.shape[0], 1))
X = np.hstack((X, tmp))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LR(max_iter=1000, learning_rate=0.1, p = 0.5)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Test score is: ", score)

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], label = "0")
plt.scatter(X[50:100, 0], X[50:100, 1], label = "1")
x = np.linspace(4, 7, 10)
plt.plot(x, (lr.odds-lr.weights[2] - lr.weights[0] * x)/lr.weights[1])
plt.legend()
plt.show()