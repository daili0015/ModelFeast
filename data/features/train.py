#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

# make pandas data for adboost train
import sys, os
sys.path.append("/home/DL/ModelFeast")

import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score


def get_data():
    data1 = np.load('train1.npy')
    data2 = np.load('train2.npy')
    data = np.concatenate((data1, data2), axis=0)
    data_X = data[:, 0:-1]
    data_y = data[:, -1]
    x_mean = data_X.mean(axis=0, keepdims=True)
    x_std = np.std(data_X, axis=0, keepdims=True)
    np.save('Xmean.npy', x_mean)
    np.save('Xstd.npy', x_std)
    data_X = (data_X-x_mean)/x_std
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, \
        test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data()


estimator = lgb.LGBMClassifier(num_leaves=31, objective='binary')
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

num_iterations = 100
#在此基础上，增加必要的参数
param = {}
param['num_leaves'] = 31
param['n_estimators'] = 100
param['objective'] = 'binary'
param['boosting'] = 'gbdt'
param['eval_metric'] = 'logloss'

model = lgb.train(param, train_data, num_iterations, \
     valid_sets=[test_data], early_stopping_rounds=50)

ypred = model.predict(X_test, num_iteration=model.best_iteration)
ypred = ypred>0.5
f1 = f1_score(ypred, y_test, average='macro')
acc = accuracy_score(ypred, y_test)
print(f1, acc)

model.save_model('model.txt')
