#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 09:55:58


import numpy as np 
import os, cv2
from SLIC import spx_img
import sys
sys.path.append('/home/DL/ModelFeast/data_preprocess')
from load import load_from_folder
from fat import fat_seg
from util import mk_dir

def spx_normalize_hu(image, masks):
    # MIN_BOUND = -115
    # MAX_BOUND = 235    
    MIN_BOUND = -400
    MAX_BOUND = 1000
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image[masks==0] = 0.

    return image.astype(np.float32)

def img_normalize_hu(image, masks):
    MIN_BOUND = -115
    MAX_BOUND = 235    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image[masks==0] = 0.

    return (image*255).astype(np.uint8)


def process_folder(dcm_dataset, crop_dataset, out_folder, fname):
    dcm_folder = os.path.join(dcm_dataset, fname)
    info_folder = os.path.join(crop_dataset, fname)
    data, masks, label = load_from_folder(info_folder=info_folder, dcm_folder=dcm_folder)

    for ind in range(data.shape[0]):

        # opencv img
        img = img_normalize_hu(data[ind], masks[ind])

        # spx
        spx_data = spx_normalize_hu(data[ind], masks[ind])*255
        slic = spx_img(spx_data, masks[ind], nr_superpixels=200)
        spx_show = slic.img_show
        spx_path = os.path.join(out_folder, str(ind)+"spx.png")
        cv2.imwrite(spx_path, spx_show)

        # fat
        fat_mask = fat_seg(data[ind], masks[ind])
        fat_path = os.path.join(out_folder, str(ind)+"fat.png")
        cv2.imwrite(fat_path, (fat_mask*255).astype(np.uint8))

        # spx-fat seg
        fat_spx_mask =  slic.mask_based_spx(fat_mask)
        fat_spx_mask = ~fat_spx_mask
        fat_spx_path = os.path.join(out_folder, str(ind)+"fat_spx.png")
        cv2.imwrite(fat_spx_path, img*fat_spx_mask)
        # break




if __name__ == '__main__':

    dcm_dataset = '/home/DL/ModelFeast/data/train_dataset'
    crop_dataset = '/SSD/data/train_cropset'
    out_folder = './tmpset'
    fname = 'E73D2FD0-21E4-4BFF-8C61-E58AA8038391'

    process_folder(dcm_dataset, crop_dataset, out_folder, fname)

    # data = normalize_hu(data)
    # data = data*masks

