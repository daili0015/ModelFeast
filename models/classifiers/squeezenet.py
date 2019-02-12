#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-09 17:29:16
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:26

import logging  # 引入logging模块
import os
import torch
from torch import load as TorchLoad
import torch.utils.model_zoo as model_zoo
from models.classifiers.Squeezenet_module import SqueezeNet


__all__ = ['squeezenet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


model_names = {
    'squeezenet1_0': 'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'squeezenet1_1-f364aa15.pth',
}

def get_squeezenet(param, pretrained = False, pretrained_path="./pretrained/"):
    
    ''' param['model_url']: download url
        param['model_version']: model_version
        param['file_name']: model file's name
        param['n_class']: how many classes to be classified
        param['img_size']: img_size, a tuple(height, width) or int
    '''

    if isinstance(param['img_size'], (tuple, list)):
        h, w = param['img_size'][0], param['img_size'][1]
    else:
        h = w = param['img_size']

    #先创建一个跟预训练模型一样结构的，方便导入权重
    model = SqueezeNet(param['model_version'], num_classes=1000)
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
    model.adaptive_set_classifier(param['n_class'])

    return model

def squeezenet(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters 
    """
    return squeezenet1_1(n_class, img_size, pretrained, pretrained_path)


def squeezenet1_0(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters 
    """
    param = {'model_url': model_urls['squeezenet1_0'], 'file_name': model_names['squeezenet1_0'], 
    'model_version': 1.0, 'n_class': n_class,  'img_size': img_size }
    return get_squeezenet(param, pretrained, pretrained_path)  


def squeezenet1_1(n_class, img_size=(224, 224), pretrained=False, pretrained_path="./pretrained/"):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters 
    """
    param = {'model_url': model_urls['squeezenet1_1'], 'file_name': model_names['squeezenet1_1'], 
    'model_version': 1.1, 'n_class': n_class,  'img_size': img_size }
    return get_squeezenet(param, pretrained, pretrained_path)  
    


if __name__ == '__main__':
    a = 7
    img_size=(a, a)
    model = squeezenet1_1()
    y = net((torch.randn(3, 3, img_size[0], img_size[1])))
    print(y.size())
