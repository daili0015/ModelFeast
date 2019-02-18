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
from helper import get_pixels_hu, normalize_hu, rescale_images


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
    data_path = os.path.join(out_dir, "ori_data.npy")
    if os.path.exists(data_path):
        pass
        # return
    slices = []
    for s in os.listdir(in_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(in_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))

    image = get_pixels_hu(slices)

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    image = rescale_images(image, pixel_spacing, 1.5, verbose=verbose)

    # image = resize_np(image, desire_dim, desire_size, verbose=verbose)

    image = normalize_hu(image)

    # print(image.dtype, image.max(), image.min()) #should be float32 between [0.0, 1.0]
    
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
        dcm2png_dir(old_folder, new_folder, desire_dim, desire_size, \
            verbose=True)

        if cnt>5: return 

train = 0
if train:
    from_dir = "./data/train_dataset"
    to_dir = "./data/train_imgset"
else:
    from_dir = "./data/test_dataset"
    to_dir = "./data/test_imgset"



from_dir = "./data/train_dataset"
to_dir = "./data/train_imgset"
process_dataset(from_dir, to_dir, 96, (128, 128))

from_dir = "./data/test_dataset"
to_dir = "./data/test_imgset"   
process_dataset(from_dir, to_dir, 96, (128, 128))

# process_dataset(from_dir, to_dir, 96, (128, 128))
# process_dataset(from_dir, "./data/tmp_set", 96, (128, 128))
# dcm2png_dir("./data/test_dataset/0A8C06EE-5B19-4B53-BBFF-DB33C495DAC9", \
#         "./data/tmp1_png", 30, (256, 256), True)
