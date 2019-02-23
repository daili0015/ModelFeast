#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import pandas as pd 
import numpy as np
import os, cv2

def get_cropsize(images_dir):
    mask = cv2.imread(os.path.join(images_dir, 'maxseg.png'), \
        cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    _, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY) 
    _, contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    assert w*h>200
    return w, h

def process_dataset(datafolder):
    folder_list = os.listdir(datafolder)
    df = pd. DataFrame(columns=['w', 'h', 'area', 'ratio'])
    cnt = 0
    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        w, h = get_cropsize(old_folder)
        res = {'w':w, 'h':h, 'area':w*h, 'ratio':w/h}
        df = df.append( [res], ignore_index=True )
        # if cnt>10: return df
        # if cnt%100==0: print("{}/{}".format(cnt, len(folder_list)))
    return df

# df = process_dataset('/home/DL/ModelFeast/data/train2_imgset')
# for col in df.columns:
#     df[col] = df[col].astype(float)
# df.to_csv('cropsize2.csv')

df = pd.read_csv('cropsize2.csv')
print(df.describe())

# 500, 380 1.3
# 250, 190, 
# same
