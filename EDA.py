#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import pydicom
import scipy.misc
import numpy as np
from PIL import Image
import os
from util import *

# out_path = '/root/图片/content.png'
# img = Image.open(out_path)
# npimg = np.asarray(img)

# print(npimg.shape)
# print(img.getpixel((0,0)))#得到像素：

# ds = pydicom.dcmread(in_path)
# print(ds.dir())  # 查看病人所有信息字典keys
# print(ds.PatientName)  # 查看病人名字

def loadFileInformation(f1, f2):
    ds = pydicom.read_file(f1)
    ds2 = pydicom.read_file(f2)

    for key in ds.dir():
        emt = ds.data_element(key)
        if key in ds2:
            emt2 = ds2.data_element(key)
            print(key, "    ------------    ", emt, "--------",emt2, "\n")
        # print(ds.SpacingBetweenSlices) #5
        # print(ds.PatientSex) #'M'
        # print(ds.CTDIvol)
        
# SliceLocation
# SliceThickness
# WindowWidth, WindowCenter - 350, 60
# PatientSex
# PatientAge

# ImagePositionPatient? ImageOrientationPatient?

# loadFileInformation(in_path)
f1 = '/home/DL/ModelFeast/data/test_dataset/0A7FE6C1-BA87-48EA-B582-5EE5C8FBFC75/a9e676a8-5273-4959-a091-712c796dfe3f_00001.dcm'
f2 = '/home/DL/ModelFeast/data/test_dataset/0A7520F7-113A-49D8-A423-6FB5EDBCA108/73b584e1-c61c-4e8f-8984-69539fd1d074_00006.dcm'
loadFileInformation(f1, f2)
# folder = './data/0A5D6760-1730-48DB-A988-FCE0FF1D6C43'
# print(is_invert_dataset('./data/test_dataset'))
# print(process_dataset('./data/test_dataset'))
# print(a)
