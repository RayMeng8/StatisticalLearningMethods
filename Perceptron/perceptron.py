# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import  matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.w = np.zeros(len(data[0])-1)
        self.alpha = np.zeros(data.shape[0])
        self.b = 0
        self.learning_rate = learning_rate

    def init_args(self, X_train, y_train):
        self.num_samples = len(X_train)
        self.num_features = len(X_train[0])
        self.gram = np.dot(X_train, X_train.T)

    #对偶形式
    def fit2(self, X_train, y_train):
        self.init_args(X_train, y_train)
        #计算Gram矩阵
        is_wrong = False
        while not is_wrong:
            wrong_cnt = 0
            for i in range(self.num_samples):
                X = X_train[i]
                y = y_train[i]
                if y * (np.sum(self.alpha * y_train * self.gram[i]) + self.b) <= 0:
                    self.alpha[i] += self.learning_rate
                    self.b += self.learning_rate * y
                    wrong_cnt += 1
            if wrong_cnt == 0:
                is_wrong = True
        self.w = np.sum(self.alpha * y_train * X_train.T, axis=1)
        print(self.w)

    #原始形式
    def fit(self, X_train, y_train):
        self.init_args(X_train, y_train)
        is_wrong = False
        while not is_wrong:
            wrong_cnt = 0
            for i in range(self.num_samples):
                X = X_train[i]
                y = y_train[i]
                pred_y = np.dot(X, self.w.T) + self.b
                if y * pred_y <=0:
                    self.w += self.learning_rate*np.dot(y, X)
                    #p = np.linalg.norm(self.w, ord=2)
                    #self.w /= p
                    self.b += self.learning_rate*y
                    wrong_cnt += 1
            if wrong_cnt == 0 :
                    is_wrong = True
        print(self.w)


#使用iris数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
df['label'] = iris.target
df.label.value_counts()

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
#plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

perceptron = Perceptron(learning_rate=1)
#perceptron.fit(X, y)
perceptron.fit2(X, y)

x1 = np.arange(4, 8, 1)
y1 = -(perceptron.w[0]*x1+perceptron.b)/perceptron.w[1]
plt.plot(x1, y1)
plt.show()