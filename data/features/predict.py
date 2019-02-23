#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27


import sys, os
sys.path.append("/home/DL/ModelFeast")
import lightgbm as lgb
import numpy as np
import pandas as pd 

def load_data():
    data = np.load('test.npy')
    x_mean = np.load('Xmean.npy')
    x_std = np.load('Xstd.npy')
    data = (data-x_mean)/x_std
    return data

data_X = load_data()
model = lgb.Booster(model_file='model.txt')
fx = model.predict(data_X)
fx[fx>0.5] = 1
fx[fx<0.5] = 0
# print(fx)

res = pd.read_csv('test.csv')
res['ret'] = fx
res['ret'] = res['ret'].astype(np.int64)
print(res.head(10))
print(res.dtypes)
res.to_csv('submit.csv', index=False)
