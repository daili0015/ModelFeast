#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-12 12:48:46
# @Last Modified by:   zcy
# @Last Modified time: 2019-01-12 13:20:36

import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold


n = 3
skf = StratifiedKFold(n_splits=n, shuffle=True)
ind = 1
df = pd.DataFrame(columns=['id', 'ret'])
dataset = '/SSD/data/train_cropset2'

# generate dataframe of dataset
folders = os.listdir(dataset)
for folder in folders:
    files = os.listdir(os.path.join(dataset, folder))
    if '0.txt' in files:
        label = 0
    elif '1.txt' in files:
        label  = 1
    else:
        print("no label file")
        print(folder)
    col = {'id':folder, 'ret':label}
    df = df.append( [col], ignore_index=True )
df['ret'] = df['ret'].astype(np.int64)
X = df['id']
y = df['ret']

for train_index, test_index in skf.split(X, y):
    print(len(train_index), len(test_index))
    train_kfolder = pd.DataFrame({'id':X[train_index], 'ret':y[train_index] })
    test_kfolder = pd.DataFrame({'id':X[test_index], 'ret':y[test_index] })
    train_kfolder.to_csv('../data/ksplit2/train{}.csv'.format(ind))
    test_kfolder.to_csv('../data/ksplit2/test{}.csv'.format(ind))
    ind += 1





# n = 3
# skf = StratifiedKFold(n_splits=n, shuffle=True)
# df = pd.read_csv('train2_label.csv')
# X = df['id']
# y = df['ret']
# ind = 1
# for train_index, test_index in skf.split(X, y):
#     print(len(train_index), len(test_index))
#     train_kfolder = pd.DataFrame({'id':X[train_index], 'ret':y[train_index] })
#     test_kfolder = pd.DataFrame({'id':X[test_index], 'ret':y[test_index] })
#     train_kfolder.to_csv('./ksplit2/train{}.csv'.format(ind))
#     test_kfolder.to_csv('./ksplit2/test{}.csv'.format(ind))
#     ind += 1

