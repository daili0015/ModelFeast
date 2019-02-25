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
    rot90_list = list()
    is_save = False
    Rot_dir = 0
    for i in range(image.shape[0]):
        org_img = image[i]
        mask, rot90 = get_mask(org_img)
        masks.append(mask)
        rot90_list.append(rot90)
        if i:
            iou = cal_iou(masks[i], masks[i-1])
            ious.append(iou)

    if not all(iou>0.6 for iou in ious):
        is_save = True
        print("\nwhat is the fuuuuuuuck?", ori_np)

    rot = np.array(list(map(abs, rot90_list))).mean()
    if 0<rot<0.2:
        print("may be need rot90: ", ori_np, rot)
        is_save = True
    elif rot>=0.2:
        is_save = True
        direction = np.array(rot90_list).mean()
        print("rot90: ", ori_np, rot)
        print("direction is ", direction)
        if direction>0:
            Rot_dir=k=1
        else:
            Rot_dir=k=3
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            masks[i] = np.rot90(masks[i], k)

    # # mask images
    # for i, mask in enumerate(masks):
    #     image[i] = image[i]*mask.astype(np.float32)

    # get max mask's bbox to crop it
    max_mask = masks[0]
    for i, mask in enumerate(masks):
        max_mask = max_mask|mask

    return masks, max_mask, is_save, Rot_dir


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
    x,y,w,h = cv2.boundingRect(hull)
    # print(x,y,w,h)

    if h>w:
        rot90 = find_rot_direction(hull, image)
        pass
    else:
        rot90 = 0
    # rot90 = 1
    # rot90 如果不需要旋转则为0 否则逆时针为1 顺时针为-1
    return mask, rot90


# 确定到底是顺时针还是逆时针转90度
# hull_contour-凸包的边缘
def find_rot_direction(hull_contour, img):

    hull_M = cv2.moments(hull_contour)
    hull_cx=int(hull_M['m10']/hull_M['m00'])
    hull_cy=int(hull_M['m01']/hull_M['m00'])
    

    sum_x=sum_y=0.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x]>0:
                sum_x += x*img[y][x]
                # sum_y += y*img[y][x]
    sum_val = img.sum()
    avg_x = int(sum_x/sum_val)
    # avg_y = int(sum_y/sum_val)
    # print(hull_cx, hull_cy)
    # print("avg", avg_x, avg_y)
    # color2 = (255, 255, 255)
    # cv2.circle(img, (hull_cx, hull_cy), 8, color2, 8)
    # cv2.circle(img, (avg_x, avg_y), 8, color2, 8)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # 如果质量重心在形状重心的左边，则说明应该逆时针转一下
    if avg_x<hull_cx:
        return 1
    else:
        return -1

# dcm2png_dir('./train_imgset/FD133D0F-CE49-4FE8-9B17-6093196C62DA', \
#     './tmp_set/FD133D0F-CE49-4FE8-9B17-6093196C62DA')