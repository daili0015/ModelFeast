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

from data_loader.ct_data_loaders2d import get_CTloader2d
import models.loss as module_loss
import models.metric as module_metric
import models as model_zoo
from utils import Logger, get_instance
from trainer import Trainer
from classifier import classifier


KofNsplit = 1

clf = classifier(model='densenet201', n_classes=2, img_size=(190-5, 250-5), pretrained=False)

clf.data_loader = get_CTloader2d('/SSD/data/train_norm', \
    './data/ksplit/train{}.csv'.format(KofNsplit), \
    '/SSD/data/train2_norm', \
    './data/ksplit2/train{}.csv'.format(KofNsplit), \
    BachSize=32, train=True, num_workers=4)
clf.valid_data_loader = get_CTloader2d('/SSD/data/train_norm', \
    './data/ksplit/test{}.csv'.format(KofNsplit), \
    '/SSD/data/train2_norm', \
    './data/ksplit2/test{}.csv'.format(KofNsplit), \
    BachSize=32, train=False, num_workers=4)


clf.set_trainer(epochs=12, save_dir = "saved/", save_period=1, verbosity=2, 
        verbose_per_epoch=20, monitor = "max val_accuracy", early_stop=4,
        steps_update=1)



resume = 0
if not resume:
    clf.set_optimizer("Adam", lr=1e-4, weight_decay=3e-4)
    clf.train()
else:
    # clf.set_optimizer("SGD", lr=1e-5, weight_decay=3e-4)
    clf.set_optimizer("Adam", lr=1e-4, weight_decay=3e-4)
    # clf.set_optimizer("SGD", lr=6e-8, weight_decay=3e-5)
    clf.train_from('/home/DL/ModelFeast/saved/PreActivationResNet/0222_200844/checkpoint_best.pth')
