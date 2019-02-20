#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-20 13:00:40


import numpy as np 
import os, cv2


def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

def get_masks(image, ori_np=None):
    masks = list()
    ious = list()
    is_save = False
    for i in range(image.shape[0]):
        org_img = image[i]
        mask = get_mask(org_img)
        masks.append(mask)
        if i:
            iou = cal_iou(masks[i], masks[i-1])
            ious.append(iou)

    if not all(iou>0.6 for iou in ious):
        is_save = True
        print("\nwhat is the fuuuuuuuck?", ori_np)

    # mask images
    for i, mask in enumerate(masks):
        image[i] = image[i]*mask.astype(np.float32)

    return image, is_save


def cal_iou(mask1, mask2):
    inter = mask1&mask2
    union = mask1|mask2
    return inter.sum()/union.sum()

def get_mask(input_img):
    image = (input_img*255).astype(np.uint8)
    mask = np.zeros(input_img.shape, dtype=np.uint8)
    # print(image.dtype, np.mean(image), np.std(image))
    # print(image.dtype, image.max(), image.min()) 

    _, binary = cv2.threshold(image, 3, 255, cv2.THRESH_BINARY)  
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)

    max_ind, max_area = 0, 0
    for i in range(len(contours)):
        if max_area<cv2.contourArea(contours[i]):
            max_area = cv2.contourArea(contours[i])
            max_ind = i

    hull = cv2.convexHull(contours[max_ind])
    cv2.fillConvexPoly(mask, hull, 1)

    return mask

# dcm2png_dir('./train_imgset/FD133D0F-CE49-4FE8-9B17-6093196C62DA', \
#     './tmp_set/FD133D0F-CE49-4FE8-9B17-6093196C62DA')