# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng 

import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


#利用决策树同一个数据集
class MaxEntropy:
    def __init__(self, dataset = []):
        self.samples = dataset
        self.num_samples = 0
        self.setY = set()
        self.w = []
        self.xy = {}
        self.num_xy = 0
        self.totolSum_xy = 0
        #self.M = 0 # M = num_xy * num
        self.xy2ID = {}
        self.ID2xy = {}
        self.E_pxy = []
        self.initPara()

    def initPara(self):
        self.num_samples = len(self.samples)
        for item in self.samples:
            X = item[:-1]
            y = item[-1]
            self.setY.add(y)
            for x in X:
                if (x, y) in self.xy:
                    self.xy[(x, y)]+=1
                else:
                    self.xy[(x, y)]=1
                self.totolSum_xy += 1
        self.num_xy = len(self.xy)
        self.w = [0.0] * self.num_xy
        id = 0
        for item in self.xy:
            self.xy2ID[item] = id
            self.ID2xy[id] = item
            id+=1
        self.E_pxy = [0.0] * len(self.xy)
        self.calc_E_pxy()

    #理解E_pxy
    def calc_E_pxy(self):
        for item in self.xy:
            id = self.xy2ID[item]
            self.E_pxy[id] = self.xy[item]/self.num_samples

    #对一个单样本
    def calc_Zx(self, X):
        Z = 0.0
        for y in self.setY:
            sum = 0.0
            for x in X:
                if (x, y) in self.xy:
                    id = self.xy2ID[(x, y)]
                    sum += self.w[id]
            Z += np.exp(sum)
        return Z

    def calc_Pyx(self, X, y):
        Zx = self.calc_Zx(X)
        sum = 0
        for x in X:
            if (x, y) in self.xy:
                id = self.xy2ID[(x, y)]
                sum += self.w[id]
        return np.exp(sum)/Zx

    def calc_Epxy(self):
        Epxy = [0.0] * self.num_xy
        for item in self.samples:
            X = item[:-1]
            y = item[-1]
            Pyx = self.calc_Pyx(X, y)
            for x in X:
                if (x, y) in self.xy:
                    id = self.xy2ID[(x, y)]
                    Epxy[id] += (1/self.num_samples)*Pyx
        return Epxy

    def train(self, iter = 100):
        for i in range(iter):
            Epxy = self.calc_Epxy()
            for (x, y) in self.xy:
                id = self.xy2ID[(x, y)]
                self.w[id] += (1.0/self.num_xy) * np.log(self.E_pxy[id]/Epxy[id])
            print("iter:" + str(i) + "\n")
            print("w:", self.w)

    def predict(self, X):
        num_Y = len(self.setY)
        pyx = 0
        res = None
        for y in self.setY:
            tmp = self.calc_Pyx(X, y)
            if tmp>pyx:
                pyx = tmp
                res = y
        return res

    def score(self, testX, testy):
        right_cnt = 0
        for i in range(len(testX)):
            if self.predict(testX[i]) == testy[i]:
                right_cnt+=1
        return right_cnt/len(testX)

datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]

me = MaxEntropy(datasets)
me.train(iter=1000)
print(me.predict(['老年', '否', '否', '一般']))
print(me.predict(['中年', '否', '是', '非常好']))