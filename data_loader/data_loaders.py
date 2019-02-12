#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-12 14:34:20

from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from base import BaseDataLoader
import numpy as np
import torch

class AutoDataLoader(BaseDataLoader):
    """automatic generate data-loader from a given folder"""
    def __init__(self, data_dir, batch_size=16, shuffle=True, validation_split=0.2, 
        num_workers=4, transform = None):
        
        if isinstance(transform, (tuple, list, int)):
            if isinstance(transform, int):
                h = w = transform
            else:
                h, w = transform[0], transform[1]
            transform = T.Compose([ 
                T.Resize(size=(h, w)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
                ])
        elif not transform:
            # default transform
            transform = T.Compose([ 
                T.Resize(size=(h, w)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
                ])
        elif callable(transform):
            pass
        else:
            print("transform 设置错误. 允许的输入为tuple, list, int, 或者任意可调用对象")
            print("输入限制都这么小了还能输错, 别玩人工智能了, 玩人工智障去吧")
            raise Exception("wrong transform type")

        self.data_dir = data_dir
        self.dataset  = ImageFolder(self.data_dir, transform = transform)
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

        super(AutoDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split,
         num_workers)
        

class CIFAR10DataLoader(BaseDataLoader):
    """docstring for CIFAR10DataLoader"""

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True,
         transform=self._tansform_)
        super(CIFAR10DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def _tansform_(self, x):
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5 
        x = x.transpose((2, 0, 1)) # 将 channel 放到第0维，这是 pytorch 要求的输入方式
        x = torch.from_numpy(x)

        # # for inceptionresnetV2
        # x = TF.to_pil_image(x)
        # x = TF.resize(x, (64, 32))
        # x = TF.to_tensor(x)

        return x
        

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        