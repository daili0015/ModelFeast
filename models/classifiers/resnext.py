#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-10 12:44:46
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:19

import logging  # 引入logging模块
import torch, os
import torch.nn as nn
from torch import load as TorchLoad
import torch.utils.model_zoo as model_zoo
from models.classifiers.ResNext101_module import resnext101_32x4d_features
from models.classifiers.ResNext101_module2 import resnext101_64x4d_features

from base import BaseModel


__all__ = ['resnext', 'resnext101_32x4d', 'resnext101_64x4d']


model_urls = {
    'resnext101_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
    'resnext101_64x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
}

model_names = {
    'resnext101_32x4d': 'resnext101_32x4d-29e315fa.pth',
    'resnext101_64x4d': 'resnext101_64x4d-e77a0586.pth',
}


class ResNeXt101_32x4d(BaseModel):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        super(ResNeXt101_32x4d, self).isValidSize(x) #check the input size

        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    # 自动调整全连接层
    def adaptive_set_fc(self, n_class):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(2048, n_class)


class ResNeXt101_64x4d(BaseModel):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        super(ResNeXt101_64x4d, self).isValidSize(x) #check the input size

        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)        
        return x

    # 自动调整全连接层
    def adaptive_set_fc(self, n_class):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(2048, n_class)


def get_resnext(param, pretrained = False, pretrained_path="./pretrained/"):

    r''' param['model_url']: download url
        param['file_name']: model file's name
        param['model_name']: model file's name
        param['n_class']: how many classes to be classified
        param['img_size']: img_size, a tuple(height, width)
    '''

    if isinstance(param['img_size'], (tuple, list)):
        h, w = param['img_size'][0], param['img_size'][1]
    else:
        h = w = param['img_size']
    # assert h>74 and w>74, 'image size should >= 75 !!!'

    #先创建一个跟预训练模型一样结构的，方便导入权重
    if param['model_name']=='resnext101_32x4d':
        model = ResNeXt101_32x4d(num_classes=1000)
    elif param['model_name']=='resnext101_64x4d':
        model = ResNeXt101_64x4d(num_classes=1000)
    model.img_size = (h, w)

    # 导入预训练模型的权值，预训练模型必须放在pretrained_path里
    if pretrained:
        if os.path.exists(os.path.join(pretrained_path, param['file_name'])):
            model.load_state_dict(TorchLoad(os.path.join(pretrained_path, param['file_name'])))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！载入权重！！")
        else:
            logging.info("pretrained 文件夹下没有，从网上下载 !!")
            model.load_state_dict(model_zoo.load_url(param['model_url'], model_dir = pretrained_path))
            logging.info("下载完毕！！载入权重！！")

    # 根据输入图像大小和类别数，自动调整
    model.adaptive_set_fc(param['n_class'])

    return model



def resnext(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    return resnext101_32x4d(n_class, img_size, pretrained, pretrained_path)

def resnext101_32x4d(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnext101_32x4d'], 'file_name': model_names['resnext101_32x4d'], 
    'model_name': 'resnext101_32x4d', 'n_class': n_class,  'img_size': img_size }
    return get_resnext(param, pretrained, pretrained_path)

def resnext101_64x4d(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnext101_64x4d'], 'file_name': model_names['resnext101_64x4d'], 
    'model_name': 'resnext101_64x4d', 'n_class': n_class,  'img_size': img_size }
    return get_resnext(param, pretrained, pretrained_path)

