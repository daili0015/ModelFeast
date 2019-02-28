#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 15:00:10
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 15:07:57



from models.StereoCNN.I3D_module import *


__all__ = [
    'i3d50', 'i3d101', 'i3d152', 
]

def i3d50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def i3d101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = I3DResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def i3d152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = I3DResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
