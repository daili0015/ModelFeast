#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-13 16:54:20
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:32

import logging  # 引入logging模块
import os
import torch.utils.model_zoo as model_zoo
from torch import load as TorchLoad
from models.classifiers.Vgg_module import vgg_Net

# python语法：限制允许从本模块导入的东西
__all__ = [ 'vgg11',  'vgg13', 'vgg16', 'vgg19' ]

# 预训练模型的下载地址
# 只要带有bn的
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_names = {
    'vgg11': 'vgg11_bn-6002323d.pth',
    'vgg13': 'vgg13_bn-abd245e5.pth',
    'vgg16': 'vgg16_bn-6c64b313.pth',
    'vgg19': 'vgg19_bn-c79401a0.pth',
}

'''
记录了不同的vgg结构
number: convolution layer's output channel
'M': MaxPool layer
'''
vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def isValidParam(param):
    '''
    check param is valid or not
    '''
    if param['img_height']%32!=0:
        logging.info("图片长宽应该设定为32的倍数，比如32 64 512这样的值，可是现在却是 %d " %param['img_height'])
        logging.error("images height must be 32's multiple, but you set it to %d " %param['img_height'])
        return False
    if param['img_width']%32!=0:
        logging.info("图片长宽应该设定为32的倍数，比如32 64 512这样的值，可是现在却是 %d " %param['img_width'])
        logging.error(" images width must be 32's multiple, but you set it to %d " %param['img_width'])
        return False
    return True

def check_param(param):
    if not isValidParam(param):
        raise RuntimeError('Error in parameter setting')

def get_vgg(Net_cfg, Net_urls, file_name, n_class, pretrained=False,
            img_size=(224, 224), pretrained_path="./pretrained/"):
    '''
    Net_cfg：网络结构
    Net_urls：预训练模型的url
    file_name：预训练模型的名字
    n_class：输出类别
    pretrained：是否使用预训练模型

    param为字典，包含网络需要的参数
    param['img_height']: image's height, must be 32's multiple
    param['img_width']: image's weight, must be 32's multiple
    '''
    if isinstance(img_size, (tuple, list)):
        h, w = img_size[0], img_size[1]
    else:
        h = w = img_size

    param = {'img_height':h, 'img_width':w}
    check_param(param)

    model = vgg_Net(Net_cfg, param) #先建立一个跟预训练模型一样的网络
    model.img_size = (h, w)
    
    if pretrained:
        if os.path.exists(os.path.join(pretrained_path, file_name)):
            model.load_state_dict(TorchLoad(os.path.join(pretrained_path, file_name)))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！直接载入！！")
        else:
            logging.info("pretrained 文件夹下没有，从网上下载 !!")
            model.load_state_dict(model_zoo.load_url(Net_urls, model_dir = pretrained_path))
            logging.info("下载完毕！！载入权重！！")

    model.adjust_classifier(n_class) #调整全连接层，迁移学习

    return model

def vgg11(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    return get_vgg(vgg_cfg['A'], model_urls['vgg11'], model_names['vgg11'], n_class, 
        pretrained, img_size, pretrained_path)

def vgg13(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    return get_vgg(vgg_cfg['B'], model_urls['vgg13'], model_names['vgg13'], n_class, 
        pretrained, img_size, pretrained_path)

def vgg16(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    return get_vgg(vgg_cfg['D'], model_urls['vgg16'], model_names['vgg16'], n_class, 
        pretrained, img_size, pretrained_path)

def vgg19(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    return get_vgg(vgg_cfg['E'], model_urls['vgg19'], model_names['vgg19'], n_class, 
        pretrained, img_size, pretrained_path)   
