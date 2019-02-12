#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-13 14:25:25
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-10 14:22:04

'''
构建vgg网络基本框架，以备后续各种vgg调用（vgg16等）
'''

import torch.nn as nn
from base import BaseModel


# python语法：限制允许从本模块导入的东西
__all__ = [ 'vgg_Net', 'adaptive_classifier' ]

'''
return a convolution layer
kernal_size, stride, and padding is always the same
'''
def get_Convlayer(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, 3, stride = 1, padding = 1)

'''
block_cfg: vgg Network's architecture
img_channel: image's channel, it's recommended to be 1 or 3
'''
# tricks： ReLU如果不inplace似乎会开辟一块新的内存，浪费空间
def construct_Conv_Block(block_cfg, img_channel):
    layer = []
    in_channel = img_channel
    for v in block_cfg:
        if v=='M':
            layer += [nn.MaxPool2d(2, stride = 2, padding = 0)]
        else:
            layer += [get_Convlayer(in_channel, v)]
            layer += [nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            in_channel = v
    return nn.Sequential(*layer) #*代表不断去除layer中的值给函数做参数

'''
the structure of classifier in pretrained model
'''
def pretrained_classifier(n_class):
    return  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_class)
        )

'''
the structure of classifier in flexible model
features//n_class < 256 : 2 layer
features//n_class > 256 : 3 layer
'''
def adaptive_classifier(features, n_class):
    layer = []
    ratio = features//n_class
    if ratio <= 256:
        h1_features = int(ratio**0.5)*n_class #hidden layer's features
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True) ]
        layer += [ nn.Linear(h1_features, n_class) ]
    else:
        cube_root = int(ratio**0.33)
        h1_features = n_class*cube_root*cube_root
        h2_features = n_class*cube_root
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True), nn.Dropout()]
        layer += [ nn.Linear(h1_features, h2_features), nn.ReLU(True), nn.Dropout() ]
        layer += [ nn.Linear(h2_features, n_class) ]

    return nn.Sequential(*layer)

class vgg_Net(BaseModel):

    def __init__(self, cfg, param):
        '''
        param为字典，包含网络需要的参数
        param['img_height']: image's height, must be 32's multiple
        param['img_width']: image's weight, must be 32's multiple
        '''
        super(vgg_Net, self).__init__()

        # features 的命名是因为要与预训练模型的命名对应，方便加载进来
        self.features = construct_Conv_Block(cfg, 3)

        conv_height, conv_width = param['img_height']//32, param['img_width']//32
        self.fc_in_features = conv_height * conv_width * 512 #全连接层的输入通道

        self.classifier = pretrained_classifier(1000)

        init_weight(self)

    def forward(self, x):

        super(vgg_Net, self).isValidSize(x) #check the input size
                
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def adjust_classifier(self, n_class):
        self.classifier = adaptive_classifier(self.fc_in_features, n_class)
        init_weight(self.classifier)
        print("网络的线性层自动调整设置为：")
        print(self.classifier)
    
def init_weight(Network):
    for m in Network.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
