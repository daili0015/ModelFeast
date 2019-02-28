#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import os, cv2
import pydicom
import numpy as np
from skimage import io

def load_from_folder(info_folder, dcm_folder):
    data = load_dcm(dcm_folder)
    masks = load_masks(info_folder)
    crop, rot_k, invert, label = load_info(info_folder)
    # print(rot_k, invert, label)
    x,y,w,h = crop
    if rot_k:
        for i in range(len(data)):
            data[i] = np.rot90(data[i], rot_k)
    if invert:
        data = data[::-1] #should be a list here

    data = np.array(data)
    data = data[:, y:y+h, x:x+w]

    return data, masks, label

def load_info(folder):
    files = os.listdir(folder)
    rot90 = 0
    label = 0
    invert = 0
    for file in files:
        if 'rot' in file:
            rot90 = int(file.split("rot")[0])
        if 'invert' in file:
            invert = 1
        if '1.txt' in file:
            label = 1
        if 'crop.npy' in file:
            crop = np.load(os.path.join(folder, 'crop.npy'))
    return crop, rot90, invert, label

def load_masks(folder):
    files = os.listdir(folder)
    masks_files = [f for f in files if 'png'in f and 'first' not in f ]
    masks_files.sort()
    masks = list()
    for m in masks_files:
        mask = io.imread(os.path.join(folder, m))
        masks.append(mask)
    masks = np.array(masks)
    masks = masks>100
    return masks 

def load_dcm(folder):
    slices = []
    for s in os.listdir(folder):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(folder, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))
    data = get_pixels_hu(slices)
    return data 

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # print(np.max(image), np.min(image))
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            print("slope != 1")
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

