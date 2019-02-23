#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

res_df = pd.read_csv('train2_submission.csv')
label_df = pd.read_csv('train2_label.csv')

suffixes=["_pred", "_true"]
merge = pd.merge(res_df, label_df, left_on="id", right_on="id", \
    suffixes=suffixes)


print(merge.head(10))


f1 = f1_score(merge['ret_pred'], merge['ret_true'], average='macro')
acc = accuracy_score(merge['ret_pred'], merge['ret_true'])
print(f1, acc)

print(merge['ret_pred'].value_counts())
print(merge['ret_true'].value_counts())
