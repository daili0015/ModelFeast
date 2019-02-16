#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import os, cv2
import pydicom
import scipy.misc
import numpy as np
from helper import get_pixels_hu, normalize_hu


def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path)) 

def resize_np(images_zyx, desire_dim, desire_size, verbose=False):
    if verbose:
        print("Shape: ", images_zyx.shape)
    # print "Resizing dim z"
    res = cv2.resize(images_zyx, dsize=(images_zyx.shape[1], desire_dim), interpolation=cv2.INTER_LINEAR)
    if verbose:
        print("after resize dim: Shape is ", res.shape)

    # resize w h 
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
    data_path = os.path.join(out_dir, "new_data.npy")
    # if os.path.exists(data_path):
    #     return
    slices = []
    for s in os.listdir(in_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(in_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))

    image = get_pixels_hu(slices)

    image = resize_np(image, desire_dim, desire_size, verbose=verbose)

    image = normalize_hu(image)

    print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    
    np.save(data_path, image) 

    # for i in range(image.shape[0]):

    #     img_path = os.path.join(out_dir, str(i).rjust(4, '0') + ".png")
    #     org_img = image[i]
        
    #     cv2.imwrite(img_path, org_img * 255)

def process_dataset(datafolder, new_datafolder, desire_dim, desire_size):
    mk_dir(new_datafolder)
    folder_list = os.listdir(datafolder)
    cnt = 0
    for folder in folder_list:
        cnt += 1
        old_folder = os.path.join(datafolder, folder)
        new_folder = os.path.join(new_datafolder, folder)
        print(str(cnt)+" : convert data from"+old_folder+"\n  to"+new_folder)
        dcm2png_dir(old_folder, new_folder, desire_dim, desire_size)

# process_dataset("./data/train_dataset", "./data/train_imgset", 30, (256, 256))
dcm2png_dir("./data/train_dataset/00CAD9C8-D90B-43C4-9964-79CAE6493DA7", \
        "./data/tmp1_png", 30, (256, 256), True)

