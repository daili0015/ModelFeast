#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 11:21:05

# dcm图像转换成为numpy数组，并且对肝脏部分进行了分割,normalize
# norm部分，归一化到0-1之间，只保留【 】之间的CT值

import os, cv2
import pydicom
import scipy.misc
import numpy as np
import pandas as pd
from util import * 
from segment import get_masks

def normalize_hu2(image):
    # MIN_BOUND = -115
    # MAX_BOUND = 235    
    MIN_BOUND = -20
    MAX_BOUND = 100   
    # image[image > MAX_BOUND] = MIN_BOUND-1
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.float32)


def dcm2png_dir(in_dir, out_dir, verbose=False):
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

    # print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    image = image*masks_np
    ori_image = ori_image*masks_np
    # for i, mask in enumerate(masks):
    #     image[i] = image[i]*mask
    #     ori_image[i] = ori_image[i]*mask
    
    # crop image
    x,y,w,h = bbox(max_mask)
    ori_image = ori_image[:, y:y+h, x:x+w]
    image = image[:, y:y+h, x:x+w]
    masks_np = masks_np[:, y:y+h, x:x+w]
    max_mask = max_mask[y:y+h, x:x+w]

    # get_bones_masks()
    # get_bones_masks(ori_image)


    # print(masks_np.dtype, masks_np.max(), masks_np.min())

    np.save(data_path, ori_image)
    np.save(seg_path, masks_np)

    cv2.imwrite(os.path.join(out_dir, 'maxseg.png'), max_mask*255) 

    is_save = 1
    if is_save:
        for i in range(image.shape[0]):
            img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
            cv2.imwrite(img_path, image[i] * 255)
            # img_path = os.path.join(out_dir, str(i).rjust(4, '0') + "c.png")
            # cv2.imwrite(img_path, ori_image[i] * 255)

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

        if cnt>17: break
        if cnt%100==0: print("{}/{}".format(cnt, len(folder_list)))


if __name__ == '__main__':

    # train = 0
    # if train==1:
    #     from_dir = "./data/train_dataset"
    #     to_dir = "./data/train_segdata"
    #     csv = "./data/train_label.csv"
    # elif train==2:
    #     from_dir = "./data/train2_dataset"
    #     to_dir = "./data/train2_segdata"
    #     csv = "./data/train2_label.csv"
    # elif train==0:
    #     from_dir = "./data/test_dataset"
    #     to_dir = "/SSD/data/test_segdata"
    #     csv = None

    # from_dir = "./data/train_dataset"
    # to_dir = "./data/train_cropset"
    # csv = "./data/train_label.csv"
    # process_dataset(from_dir, to_dir, csv)

    from_dir = "./data"
    to_dir = "./res_data"
    csv = "../train2_label.csv"
    process_dataset(from_dir, to_dir, csv)

    # folder = 'DC0792E7-1B5C-41DF-8639-1F2DBFAE2461'
    # old_folder = os.path.join(from_dir, folder)
    # new_folder = os.path.join(to_dir, folder)
    # dcm2png_dir(old_folder, new_folder, verbose=True)
