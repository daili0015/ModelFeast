#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import pydicom
import scipy.misc
import numpy as np
import pandas as pd
from PIL import Image
import os


no_age = [0]
def analyse_dcm(f):
    ds = pydicom.read_file(f)
    sex = 0 if ds.PatientSex=='M' else 1
    
    try:
        age_str = "".join(filter(str.isdigit, ds.PatientAge))
        age = float(age_str)
    except Exception as e:
        no_age[0] +=1
        age = None
    
    thickness = float(ds.SliceThickness)
    wc = float(ds.WindowCenter[0])
    ww = float(ds.WindowWidth[0])
    res = {'sex':sex, 'age':age, 'thickness':thickness, 'wc':wc, 'ww':ww}
    res['Xspace'] = float(ds.PixelSpacing[0])
    res['Yspace'] = float(ds.PixelSpacing[1])
    res['XYequal'] = 1 if res['Xspace']==res['Yspace'] else 0
    return res

def analyse_dataset(train=0):
    if train:
        root = './data/train_dataset'
        csv = './data/trainset.csv' 
    else:
        root = './data/test_dataset'
        csv = './data/testset.csv' 
    folders = os.listdir(root)
    df = pd.DataFrame(columns=['id', 'sex', 'age', 'thickness', 'wc', 'ww'] )
    cnt = 0
    for folder in folders:
        cnt += 1
        if cnt%30==0: print(cnt)

        files = os.listdir(os.path.join(root, folder))
        res = analyse_dcm( os.path.join(root, folder+"/"+files[0]) )
        res['id'] = folder
        res['cnt'] = len(files)
        res['length'] = res['cnt']*res['thickness']
        df = df.append( [res], ignore_index=True )
    df.to_csv(csv)
    return df

# analyse_dcm('./data/0A3C0BD2-708C-4AD5-BDAE-9100AD9248CC/6fa81381-7ae6-4974-b4e0-b7d933e8bf34_00001.dcm')    
# analyse_dataset(0)
# print(no_age)

# test = pd.read_csv('trainset.csv')
# test['sex'] = test['sex'].astype(int)
# print(test['sex'].value_counts())
# print(test.head())


# train = pd.read_csv('./data/trainset.csv')
# train.describe()

test = pd.read_csv('./data/testset.csv')
print(test.describe())

# print(test['sex'].value_counts())
# print(train['sex'].value_counts())
