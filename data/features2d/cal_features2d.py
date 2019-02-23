#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

import sys, os
sys.path.append("/home/DL/ModelFeast/data/")
sys.path.append("/home/DL/ModelFeast/")
import pandas as pd 
import numpy as np  
from models import densenet201 as model_arch
from util2d import load_model, get_testloader2d
import torch

model_path = '/home/DL/ModelFeast/saved/densenet201/0223_135156/checkpoint_best.pth'
# datadir = '/SSD/data/test_norm'
datadir = '/SSD/data/train_norm'

print("modules has been loaded...")
# load model
model = model_arch(n_class=2, img_size=(190-5, 250-5))

load_model(model, model_path)
print("model has been loaded...")
# dataloader
testloader = get_testloader2d(datadir, BachSize=32)
print("testloader has been loaded ...")


rets = list()
ids = list()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
print("begin to evaluate now ...")
cnt = 0
batches = len(testloader)


with torch.no_grad():
    model.eval()
    for img, fname in testloader:
        cnt+=1
        img = img.to(device)
        out = model.cal_features(img).cpu().data.numpy()
        # save to local
        for i in range(len(fname)):
            f_path = os.path.join(datadir, fname[i], 'feature.npy')
            np.save(f_path, out[i])
        #     print(out.shape)
        # break
        if cnt%10==0: print("progress {}/{} ".format(cnt, batches))

