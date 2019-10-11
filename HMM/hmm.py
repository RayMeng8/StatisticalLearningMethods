# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HMM:
    def __init__(self):
        self.I = None
        self.alpha = None
        self.beta = None

    def init_args(self, Q, V, Pi, A, B, O):
        self.N = len(Q)
        self.M = len(V)
        self.Q = Q
        self.V = V
        self.Pi = Pi
        self.A = A
        self.B = B
        self.O =  O
        self.T = len(O)
        self.I = np.array([0.0] * self.T)
        self.alpha = np.zeros((self.T, self.N))
        self.beta = np.zeros((self.T, self.N))

    def forward(self):
        for t in range(self.T):
            index = self.V.index(self.O[t])
            for i in range(self.N):
                if t==0:
                    self.alpha[t][i] = self.Pi[i] * self.B[i][index]
                else:
                    self.alpha[t][i] = np.sum([ self.alpha[t-1][k] * self.A[k][i] for k in range(self.N)]) * self.B[i][index]
        p = np.sum(self.alpha[self.T-1])
        print(p)
        return p

    def backward(self):
        for t in range(self.T-1, -1, -1):
            for i in range(self.N):
                if t==self.T-1:
                    self.beta[t][i] = 1
                else:
                    index = self.V.index(self.O[t+1])
                    self.beta[t][i] = np.sum([self.A[i][j] * self.B[j][index] * self.beta[t+1][j] for j in range(self.N)])
        index = self.V.index(self.O[0])
        p = np.sum([self.Pi[i] * self.B[i][index] * self.beta[0][i] for i in range(self.N)])
        print(p)
        return p

    def EM(self, init_Pi, init_A, init_B, max_iter):
        self.Pi = init_Pi
        self.A = init_A
        self.B = init_B
        self.forward()
        self.backward()
        eta = np.zeros((self.T, self.N))
        kusso = np.zeros((self.T, self.N, self.N))
        for it in range(max_iter):
            for t in range(self.T):
                sum_foreta = np.sum([self.alpha[t][j] * self.beta[t][j] for j in range(self.N)])
                arr_sum_forkusso = np.array(self.N)
                for i in range(self.N):
                    eta[t][i] = self.alpha[t][i] * self.beta[t][i] / sum_foreta
                    if t<self.T-1:
                        index = self.V.index(self.O[t+1])
                        arr_sum_forkusso[i] = np.sum([self.alpha[t][i] * self.A[i][j] * self.B[j][index] * self.beta[t+1][j] for j in range(self.N)])
                sum_forkusso = np.sum(arr_sum_forkusso)
                if t<self.T-1:
                    for i in range(self.N):
                        for j in range(self.N):
                            index = self.V.index(self.O[t+1])
                            kusso[t][i][j] = self.alpha[t][i] * self.A[i][j] * self.B[j][index] * self.beta[t+1][j] / sum_forkusso
            for i in range(self.N):
                self.Pi[i] = eta[0][i]
                for j in range(self.N):
                    self.A[i][j] = np.sum([kusso[t][i][j] for t in range(0, self.T-1)])/np.sum([eta[t][i] for t in range(0, self.T-1)])
                for k in range(self.M):
                    self.B[i][k] = np.sum([eta[t][i] if self.O[t]==self.V[k] else 0 for t in range(self.T)]) / np.sum([eta[t][i] for t in range(self.T)])
        return self.Pi, self.A, self.B

    def viterbi(self, Q, V, Pi, A, B, O):
        self.init_args(Q, V, Pi, A, B, O)
        delta = np.zeros((self.T, self.N))
        psis = np.zeros((self.T, self.N))
        for t in range(self.T):
            for i in range(self.N):
                if t==0:
                    index = self.V.index(self.O[t])
                    delta[t][i] = self.Pi[i] * self.B[i][index]
                    psis[t][i] = 0
                else:
                    index = self.V.index(self.O[t])
                    delta[t][i] = np.max([delta[t-1][j] * self.A[j][i] * self.B[i][index] for j in range(self.N)])
                    psis[t][i] = self.Q[np.argmax([delta[t-1][j] * self.A[j][i] for j in range(self.N)])]
        best_p = np.max(delta[self.T-1])
        best_i = np.zeros(self.T)
        best_i[self.T-1] = self.Q[np.argmax(delta[self.T-1])]
        for t in range(self.T-2, -1, -1):
            best_i[t] = psis[t+1][self.Q.index(best_i[t+1])]
        for index, state in enumerate(best_i):
            print("The %s state is %s" % (index, state))
        return best_p, best_i

Q = [1, 2, 3]
V = ['红', '白']
Pi = np.array([0.2, 0.4, 0.4])
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
O = np.array(['红', '白', '红'])

Hmm = HMM()
Hmm.init_args(Q, V, Pi, A, B, O)
Hmm.forward()
Hmm.backward()
Hmm.viterbi(Q, V, Pi, A, B, O)
