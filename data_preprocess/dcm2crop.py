#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-25 21:19:29

# dcm图像转换成为numpy数组，并且对肝脏部分进行了分割,normalize
# norm部分，归一化到0-1之间，只保留【 】之间的CT值

import os, cv2
import pydicom
import scipy.misc
import numpy as np
import pandas as pd
from helper import get_pixels_hu 
#get_pixels_hu:将dcm图像数据转换为CT值（单位为Hu）
from segment import get_masks
from util2 import is_invert_slices


def bbox(mask):
    _, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  
    _, contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)==1
    x,y,w,h = cv2.boundingRect(contours[0])
    return x,y,w,h

# def is_invert_folder(folder):
#     files = [f for f in os.listdir(folder) if ".dcm" in f]
#     files.sort(key=str)
#     ds1, ds2 = pydicom.read_file(os.path.join(folder, files[0])), \
#         pydicom.read_file(os.path.join(folder, files[1]))
#     invert_order = 1 if ds1.ImagePositionPatient[2] < ds2.ImagePositionPatient[2] else 0
#     return invert_order

def normalize_hu(image):
    MIN_BOUND = -115
    MAX_BOUND = 235
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.float32)

def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path)) 

def dcm2png_dir(in_dir, out_dir, verbose=False):
    mk_dir(out_dir)
    data_path = os.path.join(out_dir, "crop_data.npy")
    crop_path = os.path.join(out_dir, "crop.npy")
    save_image = None
    if os.path.exists(crop_path):
        pass
        # return
    slices = []
    for s in os.listdir(in_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(in_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))
    image = get_pixels_hu(slices)


    # process invert cases
    is_not_invert = is_invert_slices(image)
    invert_problem = False
    if is_not_invert>1.8:
        pass
    elif is_not_invert<0.8:
        is_not_invert = False
        invert_problem = True
    else:
        invert_problem = True
    if not is_not_invert :
    # if 1:
        invert_problem = 1
        image = image[::-1]
        print("invert it")
        f = open(os.path.join(out_dir, "invert.txt"), 'w')
        f.close()


    image = normalize_hu(image)
    masks, max_mask, is_save, rot_k = get_masks(image, data_path)


    # crop image
    x,y,w,h = bbox(max_mask)
    max_mask = max_mask[y:y+h, x:x+w]
    masks = np.array(masks)
    masks = masks[:, y:y+h, x:x+w]
    
    if rot_k:
        f = open(os.path.join(out_dir, str(rot_k)+"rot.txt"), 'w')
        f.close()

    if rot_k or invert_problem:
        save_image = image[0]
        save_image = save_image[y:y+h, x:x+w]
        save_image = save_image*masks[0].astype(np.float32)

    if save_image is not None:
        cv2.imwrite(os.path.join(out_dir, "first_image.png"), save_image*255)

    crop = np.array([x,y,w,h])
    np.save(crop_path, crop) 
    for i in range(masks.shape[0]):
        img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
        cv2.imwrite(img_path, masks[i] * 255)
        # img_path = os.path.join(out_dir, str(i).rjust(4, '0') + "t.png")
        # cv2.imwrite(img_path, image[i] * 255)

def process_dataset(datafolder, new_datafolder, csv=None):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0
    if csv:
        df = pd.read_csv(csv)

    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        # print(str(cnt)+" : convert data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder, verbose=True)

        if csv:
            label = df.query('id==@folder')['ret'].values[0]
            f = open(os.path.join(new_folder, str(label)+".txt"), 'w')
            f.close()
        if cnt>15: break
        if cnt%100==0: print("{}/{}".format(cnt, len(folder_list)))


if __name__ == '__main__':


    # from_dir = "/root/public_data/public_data/case1/train_dataset"
    # to_dir = "/root/norm_data/train_cropset"
    # csv = "/root/public_data/public_data/case1/train_label.csv"
    # process_dataset(from_dir, to_dir, csv)

    # from_dir = "/root/public_data/public_data/case1/test_dataset"
    # to_dir = "/root/norm_data/train_cropset2"
    # csv = "/root/public_data/public_data/case1/train2_label.csv"
    # process_dataset(from_dir, to_dir, csv)

    # from_dir = '/home/DL/ModelFeast/data/train_dataset'
    # to_dir = './tmpset'
    # csv = "../data/train_label.csv"
    # process_dataset(from_dir, to_dir, csv)


    dcm_dataset = '/home/DL/ModelFeast/data/train2_dataset'
    crop_dataset = '/SSD/data/train_cropset2'
    fname = 'DC0792E7-1B5C-41DF-8639-1F2DBFAE2461'

    
    dcm_folder = os.path.join(dcm_dataset, fname)
    crop_folder = os.path.join(crop_dataset, fname)
    dcm2png_dir(dcm_folder, crop_folder, verbose=True)
    # label
    csv = "/home/DL/ModelFeast/data/train2_label.csv"
    df = pd.read_csv(csv)
    if csv:
        label = df.query('id==@fname')['ret'].values[0]
        f = open(os.path.join(crop_folder, str(label)+".txt"), 'w')
        f.close()    