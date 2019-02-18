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
import os


import json
def loadFileInformation(f1, f2):
    ds = pydicom.read_file(f1)
    ds2 = pydicom.read_file(f2)

    for key in ds.dir():
        emt = ds.data_element(key)
        if key in ds2:
            emt2 = ds2.data_element(key)
            print(key, "    ------------    ", emt, "--------",emt2, "\n")
        # print(ds.SpacingBetweenSlices) #5
        # print(ds.PatientSex) #'M'
        # print(ds.CTDIvol)
# SliceLocation
# SliceThickness
# XRayTubeCurrent
# WindowWidth, WindowCenter = 350, 60

def get_wh_wc(file_name):
    '''return window center & window center of CT image'''
    ds = pydicom.read_file(file_name)
    WW = ds.data_element('WindowWidth')
    WC = ds.data_element('WindowCenter')
    return float(WW.value[0]), float(WC.value[0])

def process_dir(folder):
    files = os.listdir(folder)
    max_ww, max_wc = 0, 0
    for file in files:
        if '.dcm' in file:
            file_path = os.path.join(folder, file)
            ww, wc = get_wh_wc(file_path)
            max_ww = ww if max_ww<ww else max_ww
            max_wc = wc if max_wc<wc else max_wc
    return max_ww, max_wc

def process_dataset(dataset):
    folders = os.listdir(dataset)
    max_ww, max_wc = 0, 0
    cnt = 0
    for folder in folders:
        cnt += 1
        if cnt%30==0: print(cnt, max_ww, max_wc)
        folder_path = os.path.join(dataset, folder)
        ww, wc = process_dir(folder_path)
        max_ww = ww if max_ww<ww else max_ww
        max_wc = wc if max_wc<wc else max_wc
    return max_ww, max_wc


def is_invert_folder(folder):
    '''return window center & window center of CT image'''
    files = os.listdir(folder)
    ds1, ds2 = pydicom.read_file(os.path.join(folder, files[0])), \
        pydicom.read_file(os.path.join(folder, files[1]))
    invert_order = 1 if ds1.ImagePositionPatient[2] > ds1.ImagePositionPatient[2] else 0
    return invert_order

def is_invert_dataset(dataset):
    folders = os.listdir(dataset)
    invert_cnt, normal_cnt = 0, 0
    for folder in folders:
        folder_path = os.path.join(dataset, folder)
        invert = is_invert_folder(folder_path)
        if invert:
            invert_cnt += 1
        else:
            normal_cnt += 1
    return invert_cnt, normal_cnt


    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)

import math
def is_flip_folder(folder):
    '''return window center & window center of CT image'''
    files = os.listdir(folder)
    ds = pydicom.read_file(os.path.join(folder, files[0]))
    cos_value = (ds.ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)
    return cos_degree>0.0

def is_flip_dataset(dataset):
    folders = os.listdir(dataset)
    flip_cnt, not_flip_cnt = 0, 0
    for folder in folders:
        folder_path = os.path.join(dataset, folder)
        is_flip = is_flip_folder(folder_path)
        if is_flip:
            flip_cnt += 1
        else:
            not_flip_cnt += 1
    return flip_cnt, not_flip_cnt  