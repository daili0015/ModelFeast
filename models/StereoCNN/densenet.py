#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 13:02:27
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 13:07:05


from models.StereoCNN.Densenet_module import *


__all__ = [
    'densenet121_3d', 'densenet169_3d', 'densenet201_3d', 'densenet264_3d'
]


def densenet121_3d(**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model


def densenet169_3d(**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201_3d(**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264_3d(**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model
