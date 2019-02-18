#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import pandas as pd 
import os

data = pd.read_csv('./data/train_label.csv')
print(data.head(10))
print(data.dtypes)
# print(data['id'=='0AF1FAC8-2AD5-4CEC-9F3A-C737C7B82318'])
# e = data.query("id=='0AF1FAC8-2AD5-4CEC-9F3A-C737C7B82318'")
# # print(e)

print(data.ret.value_counts())

# for index, row in data.iterrows():
#     name_id = row['id']
#     txt_name = str(row['ret'])+".txt"
#     folder = os.path.join("./data/train_imgset/", name_id)
#     if os.path.exists(folder):
#         txt_path = os.path.join(folder, txt_name)
#         if not os.path.exists(txt_path):
#             os.mknod(txt_path)

# path = "./data/test_imgset/"
# folders = os.listdir(path)
# print(len(folders))
# pos = 0
# neg = 0
# cnt = 0
# for f in folders:
#     folder = os.path.join(path, f)
#     files = os.listdir(folder)
#     if "new_data.npy" not in files:
#         print(folder)
#     else:
#         cnt+=1
    # if "1.txt" in files and "0.txt" not in files:
    #     pos += 1
    #     continue
    # elif "0.txt" in files and "1.txt" not in files:
    #     neg += 1
    #     continue
    # else:
    #     print(folder)
# print("totally pos is ", pos, "neg is ", neg, "cnt is ", cnt)        