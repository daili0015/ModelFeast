#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-16 10:58:50
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-10 14:23:53

import torch.nn as nn
from base import BaseModel

__all__ = ['ResNet', 'Bottleneck', 'BasicBlock']


'''
kps为3 1 s
s为1或者2
s为1或者2：为1时尺寸不变，为2时可以缩减图像尺寸
'''
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

'''
kps为1 0 s
s为1或者2：为1时用来降维，为2时可以缩减图像尺寸
'''
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

'''
一个残差模块
expansion是Bottleneck才有的骚操作，BasicBlock没有，是先降维再升回来
是两个3*3
'''
class BasicBlock(nn.Module):
    expansion = 1 #定义在类内的变量是属于类的，可以self.expansion访问

    '''
    downsample是一个网络， 如果存在的话，可以把输入的深度增加expansion倍，尺寸减小
    '''
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''
一个可以实现Bottleneck的残差模块
先降维再升回来
当网络比较大时，后面通道会很多，3*3*输入通道*输出通道 是一个很大的参数量
所以用1*1的降维 然后接3*3，最后再用1*1升维回去
'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''
block可以是Bottleneck或者BasicBlock

对于确定的网络，其内部的残差块全部是同一类
resnet18 resnet34 全部用的BasicBlock
resnet50 resnet101 resnet152全部用的Bottleneck
'''
class ResNet(BaseModel):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        zero_init_residual = True
        self.block_expansion = block.expansion

        ####################### 以下是前面的层，#######################
        '''结构是7*7的卷积+3*3的池化，步长都是2，所以最终把原图缩小了1/4
        7*7的卷积： 假设输入长宽为x， 输出(x-7+2*3)/2+1 = (x-1)/2+1 
                   原文说是224进去，112出来，确实如此，不过除不尽，所以其实漏了一行和一列没处理，信息丢了
        3*3的池化：假设输入长宽为x， 输出(x-3+2*1)/2+1 = (x-1)/2+1 
                   原文说是112进去，56出来，确实如此，不过除不尽，也是漏了一行一列没做pool操作丢掉了
        最终，输出为原图的1/4，通道为64
        '''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ####################### 以上是前面的层 #######################

        ####################### 以下是残差块组成的的层 #######################
        '''总共4组残差块 各组残差块内部，维度一样，分别是 64 128 256 512

        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 4个残差块组之后，数据从56*56变成了7*7

        # AdaptiveAvgPool2d是说把任意输入变成设定的尺寸，这里是(1, 1)
        # 不需要设定pool的kernel之类的
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 一篇文章的结论，说是在残差块的最后一个Bn里初始化为0要好一点
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 制造残差块组
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # a.如果stride设置为2
        # b.要求本组的输出维度（planes*block.expansion） 不等于本组的接收的数据维度self.inplanes
        # 那么说明x跟out不能直接相加，必须用一个残差块把x， out变得一样的维度
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        #只用一个带downsample的残差块就行了，在这之后，x与out维度一样，就ok了
        # 也就是说，每个残差块组只有第一个残差块承担了改变维度尺寸的作用
        layers.append(block(self.inplanes, planes, stride, downsample)) 

        self.inplanes = planes * block.expansion #重新设定当前维度数，这是本组残差块最终的输出维度
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        super(ResNet, self).isValidSize(x) #check the input size

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def cal_features(self, x):

        super(ResNet, self).isValidSize(x) #check the input size

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    # 自动调整全连接层
    def adaptive_set_fc(self, n_class, h, w):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.block_expansion, n_class)     
        print("全连接层自动调整为：")
        print(self.fc)