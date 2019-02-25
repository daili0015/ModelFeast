#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 09:55:58

from skimage import io
import numpy as np 
import os, cv2
import pydicom


def spx_img(img, mask, nr_superpixels):
    step = int((img.shape[0]*img.shape[1]/nr_superpixels)**0.5)
    slic = SLIC(img, mask, step)
    slic.generateSuperPixels()
    slic.createConnectivity()
    img_show = slic.displayContours(color=255)
    return img_show
 

#读入图像并转化为float类型，用于传递给harris函数

def HOG_features(im):
    hog = cv2.HOGDescriptor()
    winStride = (8, 8)
    padding = (8, 8)
    hist = hog.compute(im, winStride, padding)
    hist = hist.reshape((-1,))
    return hist


filename = '0017.png'
img = io.imread(filename).astype(np.uint8)
lbp = HOG_features(img)
print(lbp.shape)
print(img.shape[0]*img.shape[1])