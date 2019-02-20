#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

import sys
sys.path.append("..")

import pandas as pd 
import numpy as np  
from models import densenet121_3d as model_arch
from util import load_model, get_testloader
import torch

model_path = '/home/DL/ModelFeast/saved/DenseNet/0219_142734/checkpoint_best.pth'
# sample = pd.read_csv('submit_example.csv')
# print(sample.head())
# print(sample.dtypes)


print("modules has been loaded...")
# load model
model = model_arch(n_classes=2, in_channels=1)
load_model(model, model_path)
print("model has been loaded...")
# dataloader
testloader = get_testloader('/SSD/data/test_norm', BachSize=16)
print("testloader has been loaded ...")



rets = list()
ids = list()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
print("begin to evaluate now ...")
cnt = 0
batches = len(testloader)

def get_pred(out):
    t = 0
    if out[0]>out[1]+t:
        return 0
    else:
        return 1

with torch.no_grad():
    model.eval()
    for img, fname in testloader:
        cnt+=1
        img = img.to(device)
        out = model(img).cpu().data

        pred = torch.argmax(out, dim=1).numpy()
        # pred = np.array(list(map(get_pred, out.numpy())))

        ids.extend(fname)
        rets.extend(pred.astype(np.int64))
        # print(pred, fname)
        # print(ids, rets)
        if cnt%10==0: print("progress {}/{} ".format(cnt, batches))

print("saving to csv file")
df = pd.DataFrame({'id': ids, 'ret': rets})
print(df.head())
print(df.dtypes)
print(df.ret.value_counts())
df.to_csv('submission7.csv', index=False)