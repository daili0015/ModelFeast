#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-31 21:15:37
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 15:29:28

''' image classifiers models '''
from .classifiers.densenet import *
from .classifiers.resnet import *
from .classifiers.xception import *
from .classifiers.inception import *
from .classifiers.vgg import *
from .classifiers.squeezenet import *
from .classifiers.inceptionresnetv2 import *
from .classifiers.resnext import *

from .StereoCNN.resnetv2 import *
from .StereoCNN.resnext import *
from .StereoCNN.densenet import *
from .StereoCNN.resnet import *
from .StereoCNN.wideresnet import *

__all__ = [ 
            'vgg11',  'vgg13', 'vgg16', 'vgg19' ,
            'squeezenet', 'squeezenet1_0', 'squeezenet1_1',
            'inception', 'inceptionv3', 
            'inceptionresnetv2',
            'xception',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext', 'resnext101_32x4d', 'resnext101_64x4d',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',


            'resnet18v2_3d', 'resnet34v2_3d', 'resnet50v2_3d', 'resnet101v2_3d',
            'resnet152v2_3d', 'resnet200v2_3d'

            'resnext50_3d', 'resnext101_3d', 'resnext152_3d',

            'densenet121_3d', 'densenet169_3d', 'densenet201_3d', 'densenet264_3d',

            'resnet10_3d', 'resnet18_3d', 'resnet34_3d', 
            'resnet101_3d', 'resnet152_3d', 'resnet200_3d', 

            'wideresnet50_3d',           
            ]
