#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-26 09:24:21

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch, os
import pandas as pd
from ct_augu import RandomCrop, Resize
from data_preprocess.load import load_from_folder

def normalize_hu(image):
    MIN_BOUND = -115
    MAX_BOUND = 235
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image.astype(np.float32)


class Kfolder_cloud(Dataset): 


    def __init__(self, root1, csv_path1, root2, csv_path2, dcm_root1, dcm_root2, train=True):
        
        self.root1 = root1
        self.root2 = root2
        self.dcm_root1 = dcm_root1
        self.dcm_root2 = dcm_root2
        self.fnames = list()
        self.labels = list()
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        for index, row in df1.iterrows():
            self.fnames.append( (1, row['id']) )
            self.labels.append( row['ret'] )
        for index, row in df2.iterrows():
            self.fnames.append( (2, row['id']) )
            self.labels.append( row['ret'] )
        self.istrain = train

    def __getitem__(self, index):

        fname = self.fnames[index]
        info_root = self.root1 if fname[0]==1 else self.root2
        dcm_root = self.dcm_root1 if fname[0]==1 else self.dcm_root2

        info_folder = os.path.join(info_root, fname[1])
        dcm_folder = os.path.join(dcm_root, fname[1])

        np_data, masks, label = load_from_folder(info_folder=info_folder, dcm_folder=dcm_folder)
        np_data = normalize_hu(np_data)
        np_data = np_data*masks

        np_data = Resize(np_data, size=(60, 190, 250))

        if self.istrain:
            np_data = RandomCrop(np_data, crop_pixels=5)

            random_val = (np.random.randint(0, 200)-100)/100.0
            np_data += random_val*0.02
        else:
            np_data = RandomCrop(np_data, crop_pixels=5)

 
        img = torch.from_numpy(np_data)
        img = img.unsqueeze(0)
        # (1, 80, 190, 250)  (channels, depth, h, w)

        return img, label

    def __len__(self):
        return len(self.fnames)

def get_CTloader_cloud( root1, csv_path1, root2, csv_path2, dcm_root1, dcm_root2, \
        BachSize=4, train=True, num_workers=4):

    trainset = Kfolder_cloud( root1, csv_path1, root2, csv_path2, dcm_root1, dcm_root2, \
        train=train)

    trainloader = DataLoader(trainset, batch_size = BachSize, \
        shuffle = False, num_workers = num_workers)
    trainloader.n_samples = len(trainset)
    return trainloader

