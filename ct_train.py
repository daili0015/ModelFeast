#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43


import os
import json
import argparse
import torch
from collections import Iterable

from data_loader.ct_data_loaders import CtKLoader, get_CTloader
import models.loss as module_loss
import models.metric as module_metric
import models as model_zoo
from utils import Logger, get_instance
from trainer import Trainer
from classifier import classifier


KofNsplit = 2
model = model_zoo.wideresnet50_3d(n_classes=2, in_channels=1)



clf = classifier(model=model, n_classes=2, img_size=256)


clf.data_loader = get_CTloader('./data/train_imgset', \
    './data/ksplit/train{}.csv'.format(KofNsplit), BachSize=12)
clf.valid_data_loader = get_CTloader('./data/train_imgset', \
    './data/ksplit/test{}.csv'.format(KofNsplit), BachSize=16, num_workers=4)

clf.set_trainer(epochs=8, save_dir = "saved/", save_period=1, verbosity=2, 
        verbose_per_epoch=20, monitor = "max val_accuracy", early_stop=3,
        steps_update=1)



resume = 1
if not resume:
    clf.set_optimizer("Adam", lr=1e-4, weight_decay=3e-5)
    clf.train()
else:
    clf.set_optimizer("SGD", lr=1e-4, weight_decay=3e-4)
    # clf.set_optimizer("Adam", lr=1e-4, weight_decay=3e-5)
    # clf.set_optimizer("SGD", lr=6e-8, weight_decay=3e-5)
    clf.train_from('/home/DL/ModelFeast/saved/WideResNet/0218_143029/checkpoint_best.pth')
