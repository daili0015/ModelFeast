#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-09 21:35:10
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:03


import logging  # 引入logging模块
import os
import torch
from torch import load as TorchLoad
import torch.utils.model_zoo as model_zoo
from models.classifiers.InceptionresnetV2_module import InceptionResNetV2

__all__ = ['inceptionresnetv2']

# image size should >= 75



model_urls = {
    'inceptionresnetv2': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
}

model_names = {
    'inceptionresnetv2': 'inceptionresnetv2-520b38e4.pth'
}


def get_inceptionresnetv2(param, pretrained = False, pretrained_path="./pretrained/"):

    r''' param['model_url']: download url
        param['file_name']: model file's name
        param['n_class']: how many classes to be classified
        param['img_size']: img_size, a tuple(height, width)
    '''

    if isinstance(param['img_size'], (tuple, list)):
        h, w = param['img_size'][0], param['img_size'][1]
    else:
        h = w = param['img_size']
    assert h>74 and w>74, 'image size should >= 75 !!!'

    #先创建一个跟预训练模型一样结构的，方便导入权重
    model = InceptionResNetV2(num_classes=1001)
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


def inceptionresnetv2(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    r"""inceptionresnetv2 performs a little better than inceptionV4
    """
    param = {'model_url': model_urls['inceptionresnetv2'], 'file_name': model_names['inceptionresnetv2'], 
    'n_class': n_class,  'img_size': img_size }
    return get_inceptionresnetv2(param, pretrained, pretrained_path)  

