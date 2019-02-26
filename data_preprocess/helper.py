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
import os, math

import cv2 

# width=350, center=60--> [-115, 235]
def normalize_hu(image):
    # MIN_BOUND = -1000.0
    # MAX_BOUND = 400.0

    # MIN_BOUND = -115
    # MAX_BOUND = 235
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image[image > 1] = 1.
    # image[image < 0] = 0.


    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.int16)

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


def rescale_images(images_zyx, org_spacing_xyz, target_voxel_mm, \
    verbose=False):

    if verbose:
        org_shape = images_zyx.shape

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    # print "Shape: ", res.shape
    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    assert res.shape[2] < 513
    res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape change from{} to {}".format(org_shape, res.shape))
    return res