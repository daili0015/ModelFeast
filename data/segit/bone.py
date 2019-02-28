#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 09:55:58


import numpy as np 
import os, cv2
import pydicom
from util import *
from segment import get_masks

def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

def bone_normalize_hu(image):
    # MIN_BOUND = -115
    # MAX_BOUND = 235    
    MIN_BOUND = 0
    MAX_BOUND = 1000
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image.astype(np.float32)

def fat_normalize_hu(image):
    res1 = np.zeros_like(image)
    res2 = np.zeros_like(image)
    res1[image>-120]=1
    res2[image<-5]=1
    mask = res1&res2
    # mask = ~mask.astype(np.bool)
    return mask

def fat_seg(img):
    mask = fat_normalize_hu(img)
    mask = (mask*255).astype(np.uint8)
    # cv2.imwrite('mask.png', mask)

    # kernel = np.ones((2, 2),np.uint8)
    # mask = cv2.erode(mask,kernel,iterations = 1)
    # kernel = np.ones((3, 3),np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask, None


def bone_seg(img):

    img = bone_normalize_hu(img)
    img = (img*255).astype(np.uint8)
    cv2.imwrite('img.png', img)

    mask = np.zeros(img.shape, dtype=np.uint8)
    # print(image.dtype, np.mean(image), np.std(image))
    # print(image.dtype, image.max(), image.min()) 

    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)  
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = list()
    for i in range(len(contours)):
        # hull = cv2.convexHull(contours[i])
        # cv2.fillConvexPoly(mask, hull, 255)
        area = cv2.contourArea(contours[i])
        if area<10:
            continue
        else:
            valid_contours.append(contours[i])
            cv2.fillConvexPoly(mask, contours[i], 255)

    return mask, binary


def dcm2png_dir(in_dir, out_dir):
    mk_dir(out_dir)
    data_path = os.path.join(out_dir, "crop_data.npy")
    seg_path = os.path.join(out_dir, "seg.npy")
    if os.path.exists(data_path):
        pass
        # return
    slices = []
    for s in os.listdir(in_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(in_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))

    invert = is_invert_folder(in_dir)
    if invert:
        slices = slices[::-1]
        print("invert it")

    image = get_pixels_hu(slices)
    ori_image = image.copy()

    image = normalize_hu(image)
    masks, max_mask, is_save, rot_k = get_masks(image, data_path)
    if rot_k:
        for i in range(image.shape[0]):
            ori_image[i] = np.rot90(ori_image[i], rot_k)
            f = open(os.path.join(out_dir, str(rot_k)+"rot.txt"), 'w')
            f.close()

    masks_np = np.array(masks)
    image = image*masks_np
    ori_image = ori_image*masks_np

    # crop image
    x,y,w,h = bbox(max_mask)
    ori_image = ori_image[:, y:y+h, x:x+w]
    image = image[:, y:y+h, x:x+w]
    masks_np = masks_np[:, y:y+h, x:x+w]
    max_mask = max_mask[y:y+h, x:x+w]

    # bone seg
    for i in range(ori_image.shape[0]):
        mask, binary = fat_seg(ori_image[i])
        # img_path = os.path.join(out_dir, "binary" + str(i).rjust(4, '0') + ".png")
        # cv2.imwrite(img_path, binary)
        img_path = os.path.join(out_dir, "mask" + str(i).rjust(4, '0') + ".png")
        cv2.imwrite(img_path, mask)

def process_dataset(datafolder, new_datafolder):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0

    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        print(str(cnt)+" : process data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder)

        if cnt>13: break
        if cnt%100==0: print("{}/{}".format(cnt, len(folder_list)))



if __name__ == '__main__':

    # folder = './data/0A3C0BD2-708C-4AD5-BDAE-9100AD9248CC'
    # dcm2png_dir(folder, "./tmpfolder")

    from_dir = "./data"
    to_dir = "./tmpset"
    process_dataset(from_dir, to_dir)