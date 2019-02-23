#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-18 17:33:17
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-10 14:13:11

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from base import BaseModel

__all__ = ['DenseNet']


'''
_DenseLayer是实现核心的concatenate的部分，它把每次的输入x与输出fx cat在一起
这样递归下去，最后一个_DenseLayer的输入也包含了第一个layer的输出的，当然也包含直接输入图片x
num_input_features：输入通道
growth_rate：输出通道
bn_size：bottleneck层的设置，由于输入通道太多，所以需要 bn-relu-conv 降维，这里conv是1*1的卷积核
本来输入是 k*growth_rate维，降维后变成4*growth_rate维；这里k是前面的Denselayer的层数，是很大的数

用nn.Sequential构建网络class的区别在于：
1.forward函数除非特殊情况，不用写 2.添加层用add_module('名字'，layer)
这里就是特殊情况才写的，因为要cat在一起
'''
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # 调用父类的方法很简单，在字类中super(子类, self).方法名()
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

# 用nn.Sequential构建网络class的典型，方便，不用写forward
# 其实Sequential本质上就是继承了Module,是高级封装,所以可以用add_module函数
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

# 降维+尺寸缩小一半
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(BaseModel):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block 有4个block

        num_init_features (int) - 前置卷积层出来能有多少特征
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()

        # First convolution 前置卷积层
        # 缩小1/4，如果尺寸为偶数，依然不完全匹配，会丢掉匹配不上的行列
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(30, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        # 4个dense块
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1: #如果不是最后一个Dense block，维度变为一半，尺寸变为一半
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer 按照预训练模型来
        self.classifier = nn.Linear(num_features, 1000)

        self.conv_features = num_features;

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        super(DenseNet, self).isValidSize(x) #check the input size

        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 不管conv输出多少，统一输出1*1 
        # 这是实现任意尺寸输入的关键
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    # 自动调整全连接层
    def adaptive_set_fc(self, n_class):
        self.classifier = nn.Linear(self.conv_features, n_class)
        print("全连接层自动调整为：")
        print(self.classifier)

    def cal_features(self, x):

        super(DenseNet, self).isValidSize(x) #check the input size

        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 不管conv输出多少，统一输出1*1 
        # 这是实现任意尺寸输入的关键
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        return out

if __name__ == '__main__':
    a = 64
    img_size=(a, a)
    net = densenet201(10, False)
    y = net((torch.randn(2, 3, img_size[0], img_size[1])))
    print(y.size())
