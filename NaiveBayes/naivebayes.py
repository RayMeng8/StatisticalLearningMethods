# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def __init__(self):
        self.probs = {}
        self.labels_num = {}
        self.prob_res = {}

    def init_args(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)
        self.num_features = len(X_train[0])

    def mean(self, X):
        return sum(X)/len(X)

    def std(self, X):
        return math.sqrt(sum([pow(x-self.mean(X), 2) for x in X])/len(X))

    def gaussian_distribution(self, x, mean, std):
        return (1/(math.sqrt(2 * math.pi)*std))*math.exp(-math.pow(x-mean, 2)/(2*math.pow(std, 2)))


    def fit(self, X, y):
        self.init_args(X, y)
        labels = list(set(y))
        for label in labels:
            prob_label = []
            data = np.array([X[i] for i in range(self.num_samples) if y[i]==label])
            self.labels_num[label] = len(data)
            for i in range(self.num_features):
                features = data[:, i]
                mean = self.mean(features)
                std = self.std(features)
                prob_label.append((mean, std))
            self.probs[label] = prob_label

    def predict(self, X):
        for label, paras in self.probs.items():
            self.prob_res[label] = 1
            for i in range(self.num_features):
                self.prob_res[label] *= self.gaussian_distribution(X[i], paras[i][0], paras[i][1])
        label = sorted(self.prob_res.items(), key = lambda x:x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right_cnt = 0
        for X, y in zip(X_test, y_test):
            if self.predict(X) == y:
                right_cnt += 1
        return right_cnt/float(len(X_test))


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, :])
X, y = data[:, :-1], data[:, -1]
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

naivebayes = NaiveBayes()
naivebayes.fit(X_train, y_train)
y_pred = naivebayes.predict([6,  2,  1,  2])
print("It belongs to ", y_pred)
score = naivebayes.score(x_test, y_test)
print("Test scores is", score)

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel("sepal width")
plt.plot(6, 2, 'r+', label = 'test_point')
plt.legend()
plt.show()


