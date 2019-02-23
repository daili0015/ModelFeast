#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-19 17:56:06


import numpy as np 
import os, cv2


def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

def normalize_hu(image):

    MIN_BOUND = -115
    MAX_BOUND = 235
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    # image = (image-0.195)/0.265

    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.float32)

def dcm2png_dir(in_dir, out_dir, verbose=False):
    mk_dir(out_dir)
    ori_np = os.path.join(in_dir, "ori_data.npy")
    norm_np = os.path.join(out_dir, "mask_data.npy")
    image = np.load(ori_np)
    
    image = normalize_hu(image) # to [0, 1] float32

    image, is_save = get_masks(image, ori_np)

    # print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    # print(image.dtype, np.mean(image), np.std(image))

    np.save(norm_np, image)

    if is_save:
        for i in range(image.shape[0]):
            img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
            org_img = image[i]
            cv2.imwrite(img_path, org_img * 255)

    return


def get_masks(image, ori_np=None):
    masks = list()
    ious = list()
    rot90_list = list()
    is_save = False
    for i in range(image.shape[0]):
        org_img = image[i]
        mask, rot90 = get_mask(org_img)
        masks.append(mask)
        rot90_list.append(rot90)
        if i:
            iou = cal_iou(masks[i], masks[i-1])
            ious.append(iou)

    if not all(iou>0.5 for iou in ious):
        is_save = True
        print("\nwhat is the fuuuuuuuck?", ori_np)
    if np.array(rot90_list).mean()>0.7:
        is_save = True
        print("\nrot90: ", ori_np)
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], 1)
            masks[i] = np.rot90(masks[i], 1)
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
    x,y,w,h = cv2.boundingRect(hull)
    # print(x,y,w,h)

    if h>w:
        rot90 = 1
    else:
        rot90 = 0

    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), 2)
    seg_image = mask*image
    # cv2.imshow('image', image)
    # cv2.imshow('seg_image', seg_image)
    # cv2.imshow('binary', binary)
    # mask2 = mask*255
    # cv2.imshow('mask', mask2)
    # # cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)
    return mask, rot90

def process_dataset(datafolder, new_datafolder):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0
    n = len(folder_list)
    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        # print(str(cnt)+" : convert data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder,  \
            verbose=False)
        if cnt%100==0:
            print("{}/{}".format(cnt, n))
        # if cnt>200:
        #     return

from_dir = "./train_imgset"
to_dir = "/SSD/data/train_mask"
process_dataset(from_dir, to_dir)


# dcm2png_dir('./train_imgset/FD133D0F-CE49-4FE8-9B17-6093196C62DA', \
#     './tmp_set/FD133D0F-CE49-4FE8-9B17-6093196C62DA')