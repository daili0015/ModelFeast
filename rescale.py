#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-13 22:37:43

import pydicom
import scipy.misc
import numpy as np
from PIL import Image
import os, math, cv2


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # print(np.max(image), np.min(image))
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def load_patient(patient_dir, new_patient_dir):
    slices = []
    for s in os.listdir(patient_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(patient_dir, s)))
    slices.sort(key=lambda x: int(x.InstanceNumber))

    for i in range(image.shape[0]):
        if not os.path.exists(new_patient_dir):
            os.mkdir(new_patient_dir)

        img_path = os.path.join(new_patient_dir, str(i).rjust(4, '0') + ".png")
        org_img = image[i]
        
        # if there exists slope,rotation image with corresponding degree
        if cos_degree>0.0:
            org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
        # cv2.imwrite(img_path, org_img * 255)


load_patient("./data/train_dataset/0A1525D9-78B4-41E7-901C-878DB83507E7", \
    "./data/tmp1")