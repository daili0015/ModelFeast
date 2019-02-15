#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 13:01:05
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-15 14:53:53

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


__all__ = ['DenseNet']



class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate,
                 block_config,
                 bn_size=4,
                 drop_rate=0,
                 n_classes=10,
                 in_channels=3):

        super(DenseNet, self).__init__()

        num_init_features=64 if in_channels==3 else 32
        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     in_channels,
                     num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, n_classes)

    def forward(self, x):
        print(x.shape)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # last_duration = int(math.ceil(self.sample_duration / 16))
        # last_size = int(math.floor(self.sample_size / 32))
        # out = F.avg_pool3d(
        #     out, kernel_size=(last_duration, last_size, last_size)).view(
        #         features.size(0), -1)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)

        out = self.classifier(out)
        return out



def densenet121_3d(**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model

if __name__ == '__main__':
    a = 64
    img_size=(a, a)
    model = densenet121_3d(n_classes=2, in_channels=1)
    x = torch.randn(3, 1, 22, img_size[0], img_size[1])
    # (BatchSize, channels, depth, h, w)
    y = model(x)
    print(y.size())

