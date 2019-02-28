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
import os, json, cv2


def normalize_hu(image):
    MIN_BOUND = 0
    MAX_BOUND = 1000
    image[image < MIN_BOUND] = MIN_BOUND
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image.astype(np.float32)


def is_invert_slices(slices):
    slices = np.array(slices)
    slices = normalize_hu(slices)
    masks = list()
    areas = list()
    for i in range(slices.shape[0]):
        mask, area = get_mask(slices[i])
        masks.append(mask)
        areas.append(area)
    N = slices.shape[0]
    np_areas = np.array(areas) 
    # np_areas = (np_areas-np_areas.min())/(np_areas.max()- np_areas.min())
    front = np.sum(np_areas[0:int(N*0.25)])
    behind = np.sum(np_areas[int(N*0.75):])

    # print(front, behind, front/behind, "sum is ", len(areas)) 
    is_not_invert = front/behind
    return is_not_invert



def get_mask(input_img):
    image = (input_img*255).astype(np.uint8)
    mask = np.zeros(input_img.shape, dtype=np.uint8)
    # print(image.dtype, np.mean(image), np.std(image))
    # print(image.dtype, image.max(), image.min()) 

    _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)  
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)

    outer_ring = cv2.convexHull(contours[0])

    for i in range(len(contours)):
        
        hull = cv2.convexHull(contours[i])
        cv2.fillConvexPoly(mask, hull, 1)
        # outer_ring += hull
        outer_ring = np.concatenate((outer_ring, hull),axis=0) 

    outer_hull = cv2.convexHull(outer_ring)
    cv2.fillConvexPoly(mask, outer_hull, 1)

    return mask, np.sum(mask)