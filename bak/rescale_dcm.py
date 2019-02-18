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

from helper import rescale_patient_images, get_pixels_hu, normalize_hu



def load_patient(patient_dir, new_patient_dir):
    slices = []
    for s in os.listdir(patient_dir):
        if ".dcm" in s:
            slices.append(pydicom.read_file(os.path.join(patient_dir, s)))

    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        assert s.SliceThickness==slice_thickness

    print( "include", len(slices), "imgs, PixelSpacing is ", slices[0].PixelSpacing)
    print("Orientation: ", slices[0].ImageOrientationPatient)

    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)
    print(cos_degree)

    image = get_pixels_hu(slices)
    print(image.shape)

    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])
    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)

    image = rescale_patient_images(image, pixel_spacing, 1.00)
    
    if not invert_order:
        image = np.flipud(image)

    print(np.max(image), np.min(image))
    for i in range(image.shape[0]):
        if not os.path.exists(new_patient_dir):
            os.mkdir(new_patient_dir)

        img_path = os.path.join(new_patient_dir, str(i).rjust(4, '0') + ".png")
        org_img = image[i]
        
        # if there exists slope,rotation image with corresponding degree
        if cos_degree>0.0:
            org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
        org_img = normalize_hu(org_img)
        cv2.imwrite(img_path, org_img * 255)


load_patient("./data/train_dataset/0A2E9075-5C56-4E0D-AA2F-383002345D7C", \
    "./data/tmp1")