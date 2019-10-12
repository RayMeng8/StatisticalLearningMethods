# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class AdaBoost1(object):
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):

        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape

        # 弱分类器数目和集合
        self.clf_sets = []

        # 初始化weights
        self.weights = [1.0 / self.M] * self.M

        # G(x)系数 alpha
        self.alpha = []

    def _G(self, features, labels, weights):
        m = len(features)
        error = 100000.0  # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i

            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])

                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])

    # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    # G(x)的线性组合
    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        self.init_args(X, y)

        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j

                # print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch, self.clf_num, j, error, best_v))
                if best_clf_error == 0:
                    break

            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)

    #             print('classifier:{}/{} error:{:.3f} v:{} direct:{} a:{:.5f}'.format(epoch+1, self.clf_num, error, best_v, final_direct, a))
    #             print('weight:{}'.format(self.weights))
    #             print('\n')

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        print(result)
        # sign
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1

        return right_count / len(X_test)




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
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

adb = AdaBoost1(n_estimators=10, learning_rate=0.2)
adb.fit(X_train, y_train)
score = adb.score(X_test, y_test)
print(score)

adb = Adaboost(num_g=10, learning_r=0.2)
adb.fit(X_train, y_train)
score = adb.score(X_test, y_test)
print(score)