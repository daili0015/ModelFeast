#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-26 09:29:43
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-26 09:37:20

import os, cv2
import pydicom
import numpy as np
from load import load_from_folder

def normalize_hu(image):
    MIN_BOUND = -115
    MAX_BOUND = 235
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image.astype(np.float32)


dataset = ''
for fodler in os.listdir(dataset):
    fodler_path = os.path.join(dataset, fodler)
    files = os.listdir(fodler_path)
    adjust_rot = False
    for f in files:
        if 'adjust' in f:
            adjust_rot = True
            rot90 = int(file.split("adjust")[0])
        if 'rot' in f:
            old_rot = int(file.split("rot")[0])
    if adjust_rot :
        print(old_rot, "adjust to ", rot90)
        if rot90!=old_rot:
            os.remove(os.path.join(fodler_path, str(old_rot)+"rot.txt"))
            f = open(os.path.join(fodler_path, str(rot90)+"rot.txt"), 'w')
            f.close()




dcm = '/home/DL/ModelFeast/data/train_dataset/DD507B2B-D6C7-49D3-B466-C84BBE038BBA'
info = '/home/DL/ModelFeast/data_preprocess/tmpset/DD507B2B-D6C7-49D3-B466-C84BBE038BBA'
data, masks, label = load_from_folder(info_folder=info, dcm_folder=dcm)
data = normalize_hu(data)
data = data*masks
