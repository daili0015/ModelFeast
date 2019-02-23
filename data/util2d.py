#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27


import numpy as np
import torch, os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ct_augu import RandomCrop, Resize


def load_model(model, resume):
    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)


class CtSet2(Dataset): 

    def __init__(self, root):
        self.root = root
        self.sample_list = os.listdir(self.root)

    def __getitem__(self, index):
        fname = self.sample_list[index]
        folder = os.path.join(self.root, fname)
        np_data = np.load(os.path.join(folder, "norm_data.npy")).astype(np.float32)

        # get 10 channels
        np_data = RandomCrop(np_data, crop_pixels=5)
        # random_val = (np.random.randint(0, 200)-100)/100.0
        # np_data += random_val*0.01
            
        img = torch.from_numpy(np_data)
        # (1, 30, 256, 256)
        # (channels, depth, h, w)

        return img, fname

    def __len__(self):
        return len(self.sample_list)

def get_testloader2d(root, BachSize=8):
    testset = CtSet2(root)
    #data/valid_data/airplane 
    testloader = DataLoader(testset, batch_size = BachSize, \
        shuffle = False, num_workers = 4)
    return testloader

