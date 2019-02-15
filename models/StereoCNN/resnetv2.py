#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-14 19:29:27
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 12:58:18


from models.StereoCNN.Resnetv2_module import *


__all__ = [
    'resnet18v2_3d', 'resnet34v2_3d', 'resnet50v2_3d', 'resnet101v2_3d',
    'resnet152v2_3d', 'resnet200v2_3d'
]


def resnet18v2_3d(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34v2_3d(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50v2_3d(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101v2_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 23, 3],
                                **kwargs)
    return model


def resnet152v2_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 8, 36, 3],
                                **kwargs)
    return model


def resnet200v2_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 24, 36, 3],
                                **kwargs)
    return model

