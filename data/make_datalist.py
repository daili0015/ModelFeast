#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-12 12:48:46
# @Last Modified by:   zcy
# @Last Modified time: 2019-01-12 13:20:36

import pandas as pd 
from sklearn.model_selection import StratifiedKFold

n = 3
skf = StratifiedKFold(n_splits=n, shuffle=True)
df = pd.read_csv('train_label.csv')
X = df['id']
y = df['ret']
ind = 1
for train_index, test_index in skf.split(X, y):
    print(len(train_index), len(test_index))
    train_kfolder = pd.DataFrame({'id':X[train_index], 'ret':y[train_index] })
    test_kfolder = pd.DataFrame({'id':X[test_index], 'ret':y[test_index] })
    train_kfolder.to_csv('./ksplit/train{}.csv'.format(ind))
    test_kfolder.to_csv('./ksplit/test{}.csv'.format(ind))
    ind += 1




