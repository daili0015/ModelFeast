#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-18 19:49:08
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-18 20:15:44

# 分析np数据  统计信息生成csv文件

import pandas as pd 
import numpy as np 
import os


def generate_csv(dataset):
    folders = os.listdir(dataset)
    df = pd.DataFrame(columns=['Z', 'Y', 'X'])
    n = len(folders)
    for i, folder in enumerate(folders):
        np_file = os.path.join(dataset, folder+"/crop_data.npy")
        data = np.load(np_file)
        res = {'Z':data.shape[0], 'Y':data.shape[1], 'X':data.shape[2]}
        res['Z/Y'] = data.shape[0]/data.shape[1]
        res['X/Y'] = data.shape[2]/data.shape[1]
        # res['volume'] = int(data.shape[0]*data.shape[1]**2/1e6)
        # res['area'] = int(data.shape[1]**2/1e4)
        # res['mean'] = data.mean()
        # res['std'] = np.std(data)
        df = df.append( [res], ignore_index=True )
        if i%30==0 and i!=0: 
            print(" process {}/{}".format(i, n))
            # return df
    return df

train = 0

if train:
    dataset = './train_cropset'
    csv = 'train_shapes_means.csv'
else:
    dataset = './train2_cropset'
    csv = 'train2_shapes_means.csv'

# df = generate_csv(dataset)
# df.to_csv(csv)
# print(df.head())


import pandas as pd
train = pd.read_csv('train_shapes_means.csv')
print(train.describe())
test = pd.read_csv('train2_shapes_means.csv')
test.describe()



