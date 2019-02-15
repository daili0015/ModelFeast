#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 12:55:08
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 13:00:45



from models.StereoCNN.Resnext_module import *



__all__ = [
    'resnext50_3d', 'resnext101_3d', 'resnext152_3d'
]


def resnetx50_3d(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model