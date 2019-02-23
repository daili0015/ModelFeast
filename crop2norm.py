#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27


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
    image = (image-0.5)/0.5

    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.float32)

def resize_np(images_zyx, desire_dim, desire_size, verbose=False):
    if verbose:
        print("Shape: ", images_zyx.shape)
    # print "Resizing dim z"
    res = cv2.resize(images_zyx, dsize=(images_zyx.shape[1], desire_dim), interpolation=cv2.INTER_LINEAR)
    if verbose:
        print("after resize dim: Shape is ", res.shape)
    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    assert res.shape[2] < 513
    res = cv2.resize(res, dsize=desire_size, interpolation=cv2.INTER_LINEAR)
    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("after resize w and h: Shape is ", res.shape)
    return res


def dcm2png_dir(in_dir, out_dir, desire_dim, desire_size, verbose=False):
    mk_dir(out_dir)
    ori_np = os.path.join(in_dir, "crop_data.npy")
    norm_np = os.path.join(out_dir, "norm_data.npy")
    mask = np.load(ori_np.replace('crop_data', 'seg'))
    image = np.load(ori_np)

    image = resize_np(image, desire_dim, desire_size, verbose=verbose)
    image = normalize_hu(image)
    mask = cv2.resize(mask, dsize=desire_size, interpolation=cv2.INTER_LINEAR)
    image = image*mask

    # print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    
    np.save(norm_np, image) 

    # for i in range(image.shape[0]):
    #     img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
    #     org_img = image[i]
    #     cv2.imwrite(img_path, org_img * 255)
    return 


def process_dataset(datafolder, new_datafolder, desire_dim, desire_size):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0
    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        # print(str(cnt)+" : convert data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder, desire_dim, desire_size, \
            verbose=False)
        if cnt%100==0: print(cnt)
        # if cnt>2: 
        #     return 

# from_dir = "./train_cropset"
# to_dir = "/SSD/data/train_norm"
# process_dataset(from_dir, to_dir, 30, (250, 190))

# from_dir = "./train2_cropset"
# to_dir = "/SSD/data/train2_norm"
# process_dataset(from_dir, to_dir, 30, (250, 190))

from_dir = "/SSD/data/test_segdata"
to_dir = "/SSD/data/test_norm"
process_dataset(from_dir, to_dir, 30, (250, 190))

# dcm2png_dir('./train_imgset/DD507B2B-D6C7-49D3-B466-C84BBE038BBA', 
#     './tmp_set/DD507B2B-D6C7-49D3-B466-C84BBE038BBA', 
#     85, 136)