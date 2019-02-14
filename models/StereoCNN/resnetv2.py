#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-14 19:29:27
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-14 20:08:22


from models.StereoCNN.Resnetv2_module import *


__all__ = [
    'resnet18_3dv2', 'resnet34_3dv2', 'resnet50_3dv2', 'resnet101_3dv2',
    'resnet152_3dv2', 'resnet200_3dv2'
]


def resnet18_3dv2(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_3dv2(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_3dv2(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_3dv2(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 23, 3],
                                **kwargs)
    return model


def resnet152_3dv2(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 8, 36, 3],
                                **kwargs)
    return model


def resnet200_3dv2(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 24, 36, 3],
                                **kwargs)
    return model

