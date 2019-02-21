#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

# dcm图像转换成为numpy数组，并且对肝脏部分进行了分割,normalize
# norm部分，归一化到0-1之间，只保留【 】之间的CT值

import os, cv2
import pydicom
import scipy.misc
import numpy as np
from helper import get_pixels_hu 
#get_pixels_hu:将dcm图像数据转换为CT值（单位为Hu）
from segment import get_masks

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
    data_path = os.path.join(out_dir, "seg_data.npy")
    seg_path = os.path.join(out_dir, "seg.npy")
    if os.path.exists(data_path):
        pass
        # return
    slices = []
    for s in os.listdir(in_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(in_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))

    image = get_pixels_hu(slices)

    image = normalize_hu(image)

    masks, max_mask, is_save = get_masks(image, data_path)
    # print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    for i, mask in enumerate(masks):
        image[i] = image[i]*mask
    
    # print(image.dtype, image.max(), image.min())
    np.save(data_path, image.astype(np.float16))
    np.save(seg_path, max_mask)
    # print(max_mask.dtype)
    cv2.imwrite(os.path.join(out_dir, 'maxseg.png'), max_mask*255) 

    if is_save:
        # image = image.astype(np.float32)
        for i in range(image.shape[0]):
            img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
            org_img = image[i]
            cv2.imwrite(img_path, org_img * 255)


def process_dataset(datafolder, new_datafolder):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0
    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        # print(str(cnt)+" : convert data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder, verbose=True)

        # if cnt>1: return 
        if cnt%100==0: print("{}/{}".format(cnt, len(folder_list)))



if __name__ == '__main__':
    # train = 1
    # if train:
    #     from_dir = "./data/train_dataset"
    #     to_dir = "./data/train_segdata"
    # else:
    #     from_dir = "./data/test_dataset"
    #     to_dir = "./data/test_segdata"


    # from_dir = "./data/train_dataset"
    # to_dir = "./data/train_imgset"
    # process_dataset(from_dir, to_dir)

    from_dir = "./data/train2_dataset"
    to_dir = "./data/train2_imgset"
    process_dataset(from_dir, to_dir)

    # folder = 'BB79514D-6A0E-43FF-9B06-A600CD3302A8'
    # old_folder = os.path.join(from_dir, folder)
    # new_folder = os.path.join(to_dir, folder)
    # dcm2png_dir(old_folder, new_folder, verbose=True)

    # dcm2png_dir("./data/test_dataset/0A8C06EE-5B19-4B53-BBFF-DB33C495DAC9", \
    #         "./data/tmp1_png", 30, (256, 256), True)
