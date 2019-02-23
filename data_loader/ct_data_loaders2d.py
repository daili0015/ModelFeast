#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-12 14:34:20

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from base import BaseDataLoader
import numpy as np
import torch, os
import pandas as pd
from ct_augu import RandomCrop, Resize

class Kfolder2d(Dataset): 

    def __init__(self, root1, csv_path1, root2, csv_path2, train=True):
        
        self.root1 = root1
        self.root2 = root2
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
        root = self.root1 if fname[0]==1 else self.root2
        folder = os.path.join(root, fname[1])

        np_data = np.load(os.path.join(folder, "norm_data.npy"))
        # get 10 channels
        if self.istrain:
            np_data = RandomCrop(np_data, crop_pixels=5)

            random_val = (np.random.randint(0, 200)-100)/100.0
            np_data += random_val*0.02
        else:
            np_data = RandomCrop(np_data, crop_pixels=5)
            # random_val = (np.random.randint(0, 200)-100)/100.0
            # np_data += random_val*0.02
            
        img = torch.from_numpy(np_data)
        # (10, 256, 256)
        # (channels, h, w)

        return img, self.labels[index]

    def __len__(self):
        return len(self.fnames)

def get_CTloader2d( root1, csv_path1, root2, csv_path2, BachSize=4,\
         train=True, num_workers=4):

    trainset = Kfolder2d( root1, csv_path1, root2, csv_path2, \
        train=train)

    trainloader = DataLoader(trainset, batch_size = BachSize, \
        shuffle = False, num_workers = num_workers)
    trainloader.n_samples = len(trainset)
    return trainloader
