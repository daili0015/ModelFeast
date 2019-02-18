#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-16 11:33:27

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch, os

def load_model(model, resume):
    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)


class CtSet(Dataset): 

    def __init__(self, root):
        self.root = root
        self.sample_list = os.listdir(self.root)

    def __getitem__(self, index):
        fname = self.sample_list[index]
        folder = os.path.join(self.root, fname)
        np_data = np.load(os.path.join(folder, "new_data2.npy"))
        # np_data = (np_data-0.5)/0.5 # to [-1, 1]
        np_data = (np_data-0.2)/0.25 
        img = torch.from_numpy(np_data)
        img = img.unsqueeze(0)
        # (1, 30, 256, 256)
        # (channels, depth, h, w)

        return img, fname

    def __len__(self):
        return len(self.sample_list)

def get_testloader(root, BachSize=8):
    testset = CtSet(root)
    #data/valid_data/airplane 
    testloader = DataLoader(testset, batch_size = BachSize, shuffle = False, num_workers = 4)
    return testloader

