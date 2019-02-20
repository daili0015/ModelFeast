#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

import pandas as pd 
import numpy as np  

files = ['submission'+str(i)+'.csv' for i in range(1, 8)]
df = pd.read_csv(files[0])
for i, file in enumerate(files[1:]):
    suffixes=["_L"+str(i), "_R"+str(i)]
    df1 = pd.read_csv(file)
    df = pd.merge(df, df1, left_on="id", right_on="id", suffixes=suffixes)
cols = df.columns

suffixes=["_L", "_R"]

print(cols)
print(df.head())
vote_ret = list()
for index, row in df.iterrows():
    pos_vote = neg_voye = 0
    for col in cols:
        if 'ret' in col:
            if row[col]==1:
                pos_vote+=1
            else:
                neg_voye+=1
    res = 1 if pos_vote>neg_voye else 0
    vote_ret.append(res)

df['vote'] = vote_ret
print(df.head())

df_save = pd.DataFrame({'id': df['id'], 'ret': df['vote']})
df_save.to_csv('submission_vote.csv', index=False)
print(df_save.head())
print(df_save.dtypes)
print(df_save['ret'].value_counts())
