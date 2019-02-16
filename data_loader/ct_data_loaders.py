#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-12 14:34:20

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from base import BaseDataLoader
import numpy as np
import torch, os

class CtFolder(Dataset): 

    def __init__(self, root):
        self.root = root
        self.sample_list = list()
        self.labels = list()
        sample_number = 3000
        pos = neg = 0
        for d in  os.listdir(self.root):
            folder = os.path.join(self.root, d)
            file_list = os.listdir(folder)
            if "1.txt" in file_list:
                pos += 1
                if pos<=sample_number:
                    self.sample_list.append(d)
                    self.labels.append(1)
            elif "0.txt" in file_list:
                neg += 1
                if neg<=sample_number:
                    self.sample_list.append(d)
                    self.labels.append(0)
            else:
                raise Exception("no label file!", folder)


    def __getitem__(self, index):
        sampler = self.sample_list[index]
        folder = os.path.join(self.root, sampler)
        np_data = np.load(os.path.join(folder, "new_data.npy"))

        np_data = (np_data-0.5)/0.5 # to [-1, 1]

        img = torch.from_numpy(np_data)
        img = img.unsqueeze(0)
        # (1, 30, 256, 256)
        # (channels, depth, h, w)

        return img, self.labels[index]

    def __len__(self):
        return len(self.sample_list)

class CtDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, \
            training=True):
        
        self.data_dir = data_dir
        self.dataset = CtFolder(self.data_dir)

        super(CtDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        