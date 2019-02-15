#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 15:27:29
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 15:34:46



from models.StereoCNN.WideResnet_module import *


__all__ = [
    'wideresnet50_3d',
]


def wideresnet50_3d(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = WideResNet(WideBottleneck, [3, 4, 6, 3], **kwargs)
    return model