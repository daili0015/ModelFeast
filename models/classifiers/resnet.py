#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-16 12:40:14
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:12

import logging  # 引入logging模块
import torch, os
import torch.nn as nn
from torch import load as TorchLoad
import torch.utils.model_zoo as model_zoo
from models.classifiers.ResNet_module import ResNet, Bottleneck, BasicBlock


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_names = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def resnet18(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnet18'], 'file_name': model_names['resnet18'], 
    'layers': [2, 2, 2, 2], 'block': BasicBlock, 'n_class': n_class,  'img_size': img_size }
    return get_resnet(param, pretrained, pretrained_path)

def resnet34(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnet34'], 'file_name': model_names['resnet34'], 
    'layers': [3, 4, 6, 3], 'block': BasicBlock, 'n_class': n_class,  'img_size': img_size }
    return get_resnet(param, pretrained, pretrained_path)

def resnet50(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnet50'], 'file_name': model_names['resnet50'], 
    'layers': [3, 4, 6, 3], 'block': Bottleneck, 'n_class': n_class,  'img_size': img_size }
    return get_resnet(param, pretrained, pretrained_path)
    
def resnet101(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnet101'], 'file_name': model_names['resnet101'], 
    'layers': [3, 4, 23, 3], 'block': Bottleneck, 'n_class': n_class,  'img_size': img_size }
    return get_resnet(param, pretrained, pretrained_path)   

def resnet152(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    param = {'model_url': model_urls['resnet152'], 'file_name': model_names['resnet152'], 
    'layers': [3, 8, 36, 3], 'block': Bottleneck, 'n_class': n_class,  'img_size': img_size }
    return get_resnet(param, pretrained, pretrained_path)   


def get_resnet(param, pretrained = False, pretrained_path="./pretrained/"):
    
    ''' param['model_url']: download url
        param['file_name']: model file's name
        param['layers']: res block setting
        param['block']: which kind of res block to use
        param['n_class']: how many classes to be classified
        param['img_size']: img_size, a tuple(height, width)
    '''

    if isinstance(param['img_size'], (tuple, list)):
        h, w = param['img_size'][0], param['img_size'][1]
    else:
        h = w = param['img_size']

    #先创建一个跟预训练模型一样结构的，方便导入权重
    model = ResNet(param['block'], param['layers'], num_classes=1000)
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
    model.adaptive_set_fc(param['n_class'], h, w)

    return model

if __name__ == '__main__':
    a = 500
    img_size=(a, a)
    net = resnet101(10, img_size, True)
    y = net((torch.randn(3, 3, img_size[0], img_size[1])))
    print(y.size())
