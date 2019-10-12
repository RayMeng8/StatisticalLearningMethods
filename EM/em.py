# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class EMThreeCoins:
    def __init__(self, init_prob, max_iter):
        self.pi, self.p, self.q = init_prob
        self.max_iter = max_iter
        #self.data = None
        #self.u = None

    def init_args(self, data):
        self.data = np.array(data)
        self.u = np.array([0.0] * len(data))

    def _E(self):
        #计算来自B硬币的期望／分模型对观测数据的响应度
        for i in range(len(self.data)):
            prob_b = self.pi * math.pow(self.p, self.data[i]) * math.pow(1-self.p, 1-self.data[i])
            prob_c = (1-self.pi) * math.pow(self.q, self.data[i]) * math.pow(1-self.q, 1-self.data[i])
            self.u[i] = prob_b/(prob_b+prob_c)

    def _M(self, data):
        #更新参数
        self.init_args(data)
        for i in range(self.max_iter):
            self._E()
            self.pi = np.mean(self.u)
            self.p = np.sum([self.u[k] * self.data[k] for k in range(len(self.data))])/np.sum(self.u)
            self.u_q = 1 - self.u
            self.q = np.sum([self.u_q[k] * self.data[k] for k in range(len(self.data))])/np.sum(self.u_q)
            print("In iter%s: pi=%f p=%f q=%f" % (i+1, self.pi, self.p, self.q))

#对初值敏感
init_prob = [0.4, 0.6, 0.7]
emthreecoins = EMThreeCoins(init_prob=init_prob, max_iter=10)
data = [1,1,0,1,0,0,1,0,1,1]
emthreecoins._M(data)
