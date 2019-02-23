#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

# make pandas data for adboost train
import sys, os
sys.path.append("/home/DL/ModelFeast")
import pydicom
import pandas as pd 
import numpy as np  


dataset = 3
isTrain = 0
KofNsplit = 1
feature_channel = 1916

part = 'train' if isTrain else 'test'
if dataset==1:
    datadir = '/SSD/data/train_norm'
    dcmdir = '/home/DL/ModelFeast/data/train_dataset'
    csv = '/home/DL/ModelFeast/data/ksplit/{}{}.csv'.format(part, KofNsplit)
elif dataset==2:
    datadir = '/SSD/data/train2_norm'
    dcmdir = '/home/DL/ModelFeast/data/train2_dataset'
    csv = '/home/DL/ModelFeast/data/ksplit2/{}{}.csv'.format(part, KofNsplit)
elif dataset==3:
    datadir = '/SSD/data/test_norm'
    dcmdir = '/home/DL/ModelFeast/data/test_dataset'

def get_infomation(datadir, fname):
    folder = os.path.join(datadir, fname)
    files = [f for f in os.listdir(folder) if 'dcm' in f]
    ds = pydicom.read_file(os.path.join(folder, files[0]))
    # sex
    sex = 1 if 'M' in str(ds.PatientSex) else 0
    # age
    try:
        age_str = "".join(filter(str.isdigit, ds.PatientAge))
        age = float(age_str)
    except Exception as e:
        age = 50   
    return np.array([sex, age], dtype=np.float32)

if dataset!=3:
    # make train dataset
    df = pd.read_csv(csv)
    print(df.shape[0])
    n = df.shape[0]
    features = np.zeros(shape=(n, feature_channel+2+1), dtype=np.float32)
    for index, row in df.iterrows():
        fname = row['id']
        label = row['ret']
        feature = np.load(os.path.join(datadir, fname, 'feature.npy'))
        infomations = get_infomation(dcmdir, fname)
        features[index, 0:feature.shape[0]] = feature[:]
        features[index, feature.shape[0]:-1] = infomations[:]
        features[index, -1] = label
        if index%100==0: print("{}/{}".format(index, n))

    print(features.dtype, features.mean(), features.shape)
    print(features[:, -3].mean())
    print(features[:, -2].mean())
    np.save('train{}.npy'.format(dataset), features)
    
else:
    # make test dataset
    folders = os.listdir(datadir)
    n = len(folders)
    df = pd.DataFrame(columns=['id'])
    features = np.zeros(shape=(n, feature_channel+2), dtype=np.float32)
    for i, fname in enumerate(folders):
        feature = np.load(os.path.join(datadir, fname, 'feature.npy'))
        infomations = get_infomation(dcmdir, fname)
        features[i, 0:feature.shape[0]] = feature[:]
        features[i, feature.shape[0]:] = infomations[:]
        if i%100==0: print("{}/{}".format(i, n))

        res = {'id':str(fname)}
        df = df.append( [res], ignore_index=True )
        
    print(features.dtype, features.mean(), features.shape)
    np.save('test.npy'.format(dataset), features)
    df.to_csv('test.csv', index=False)
    print(df.head())


