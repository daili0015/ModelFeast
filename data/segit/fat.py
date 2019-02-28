#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 09:55:58


import numpy as np 
import os, cv2

def fat_normalize_hu(image):
    res1 = np.zeros_like(image)
    res2 = np.zeros_like(image)
    res1[image>-12000]=1
    res2[image<-5]=1
    mask = res1&res2
    # mask = ~mask.astype(np.bool)
    return mask

def fat_seg(img, crop_mask):
    mask = fat_normalize_hu(img)
    mask = crop_mask*mask
    # mask = (mask*255).astype(np.uint8)
    # cv2.imwrite('mask.png', mask)
    # kernel = np.ones((2, 2),np.uint8)
    # mask = cv2.erode(mask,kernel,iterations = 1)
    # kernel = np.ones((3, 3),np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask