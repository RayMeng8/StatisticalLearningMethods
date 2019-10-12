# !/usr/bin/env python3
# coding:utf-8
# Author:Ray.Meng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, root = True, label = None, feature_name = None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, data):
        if self.root:
            return self.label
        return self.tree[data[self.feature]].predict(data)


class DecisionTree():
    def __init__(self, lim = 0.1):
        self.tree  = {}
        self.lim = lim

    def calc_ent(self, data):
        data_len = len(data)
        label_count = {}
        for i in range(data_len):
            label = data[i][-1]
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1
        ent = 0
        for i in label_count.values():
            ent += i / data_len * np.log2(i / data_len)
        ent = -ent
        return ent

    def calc_condi_ent(self, data, axis=0):
        data_len = len(data)
        feat_set = {}
        for i in range(data_len):
            feature = data[i][axis]
            if feature not in feat_set:
                feat_set[feature] = []
            feat_set[feature].append(data[i])
        condi_ent = sum(len(p) / data_len * self.calc_ent(p) for p in feat_set.values())
        return condi_ent

    def info_gain(self, ent, condi_ent):
        return ent - condi_ent

    def info_gain_train(self, data):
        cnt = len(data[0]) - 1
        ent = self.calc_ent(data)
        best_feature = []
        for i in range(cnt):
            info_gain1 = self.info_gain(self.calc_ent(data), self.calc_condi_ent(data, i))
            best_feature.append([i, info_gain1])
            print(labels[i] + "的信息增益为" + str(info_gain1))
        best = max(best_feature, key=lambda x: x[-1])
        print("选择特征：" + str(labels[best[0]]) + "\n信息增益为：" + str(best[1]))
        return best

    def train(self, data):
        y_train, features = data.iloc[:, -1], data.columns[:-1]

        if len(y_train.value_counts()) == 1:
            return Node(True, label=y_train.iloc[0])

        if len(features) == 0:
            return Node(True, label=y_train.value_counts().sort_values(ascending=False).indexp[0])

        best_feature_no, best_info_gain = self.info_gain_train(np.array(data))
        best_feature_name = features[best_feature_no]
        if best_info_gain<self.lim:
            return Node(True, label=y_train.value_counts().sort_values(ascending=False).indexp[0])

        node = Node(False, feature=best_feature_no, feature_name=best_feature_name)
        feature_list = data.iloc[:,best_feature_no].value_counts().index
        for f in feature_list:
#???
            sub_train_data = data.loc[f == data[best_feature_name]].drop([best_feature_name], axis=1)
            sub_node = self.train(sub_train_data)
            node.add_node(f, sub_node)
        return node

    def fit(self, data):
        self.tree = self.train(data)
        return self.tree

    def predict(self, data):
        return self.tree.predict(data)


features = [['青年', '否', '否', '一般', '否'],
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
               ['老年', '否', '否', '一般', '否']]
labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']

data = pd.DataFrame(features, columns=labels)
dt = DecisionTree()
tree = dt.fit(data)
print(dt.predict(['老年', '否', '否', '一般']))
