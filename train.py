#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-03-05 12:24:17
# @Last Modified by:   zcy
# @Last Modified time: 2019-03-05 12:49:57



from classifier import classifier

if __name__ == '__main__':
    clf = classifier('xception', 17, (60, 60), 'E:/Oxford_Flowers17/train')
    clf.set_trainer(epochs=1, save_dir="saved/", save_period=1, verbosity=2, 
        verbose_per_epoch=100, monitor = "max val_accuracy", early_stop=10, 
        tensorboardX=False, log_dir="saved/runs", steps_update=1)
    clf.train()



