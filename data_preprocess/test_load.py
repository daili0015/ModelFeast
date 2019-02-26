#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

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

dcm = '/home/DL/ModelFeast/data/train_dataset/DD507B2B-D6C7-49D3-B466-C84BBE038BBA'
info = '/home/DL/ModelFeast/data_preprocess/tmpset/DD507B2B-D6C7-49D3-B466-C84BBE038BBA'
data, masks, label = load_from_folder(info_folder=info, dcm_folder=dcm)
data = normalize_hu(data)
data = data*masks

for i in range(data.shape[0]):
    img_path = os.path.join('/home/DL/ModelFeast/data_preprocess/test', \
            str(i).rjust(4, '0') + ".png")
    cv2.imwrite(img_path, data[i] * 255)

