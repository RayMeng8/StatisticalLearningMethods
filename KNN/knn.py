# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class KNN:
    def __init__(self, X_train, y_train, k=3, p=2):
        #确定K和距离度量
        self.k = k
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)
        self.num_features = len(X_train[0])

    def predict(self, X):
        arr_dist = np.zeros(self.num_samples)
        for i in range(len(self.X_train)):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            arr_dist[i] = dist
        sort_idx = np.argsort(arr_dist)
        knn = [y_train[sort_idx[i]] for i in range(self.k)]
        count_pairs = Counter(knn)
        y_pred = sorted(count_pairs.items(), key = lambda x:x[1], reverse=True)[0][0]
        return y_pred

    def score(self, X_test, y_test):
        right_cnt = 0
        for X, y in zip(X_test, y_test):
            if self.predict(X)==y:
                right_cnt+=1
        return right_cnt/len(X_test)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
df['label'] = iris.target
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNN(X_train, y_train, k=3, p=2)
score = knn.score(X_test, y_test)
print("Test score is ", score)

test_point = [6, 3.5]
print("It belongs to type ", knn.predict(test_point))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='-1')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'b+', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

