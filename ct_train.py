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

from data_loader.ct_data_loaders import CtDataLoader
import models.loss as module_loss
import models.metric as module_metric
import models as model_zoo
from utils import Logger, get_instance
from trainer import Trainer
from classifier import classifier



clf = classifier(model='squeezenet', n_classes=2, img_size=512)

clf.data_loader = CtDataLoader('./data/train_imgset', batch_size=4, shuffle=True, \
    validation_split=0.3, num_workers=4)
clf.valid_data_loader = clf.data_loader.split_validation()
clf.model = model_zoo.resnet18_3dv2(n_classes=2, in_channels=1)

clf.set_trainer(epochs=10, save_dir="saved/", save_period=1, verbosity=2, 
        verbose_per_epoch=20, monitor = "max val_accuracy", early_stop=10)

clf.set_optimizer("Adam", lr=1e-4, weight_decay=1e-4)

clf.train()

