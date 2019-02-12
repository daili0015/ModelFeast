#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-14 18:02:38
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-10 14:24:36

# 定义了inception网络中各个inception模块
# 由于Inception网络对图像尺寸限制极大，几乎其输入必须是299的，不然就得大改网络结构
# 故而这里采用原结构，只允许输出的类别数不同

import torch
import torch.nn as nn
import torch.nn.functional as F



# python语法：限制允许从本模块导入的东西
__all__ = [ 'BasicConv2d',  'InceptionA', 'InceptionB', 'InceptionC', 'InceptionD', 
        'InceptionE', 'InceptionAux', ]


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        # 按照官方设定为无偏置，需要测试下有偏置会怎样
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

'''
InceptionV3是把3*3卷积核拆解了
不过论文中提到，这种拆分在中等尺寸的特征图上才有效（保证模型准确率同时降低模型大小）
而InceptionA是前面的模块，所以没有采用这种拆分
这个模块里，除了输入的通道数，pool那条支路的输出通道也是不固定的参数，需要输入
除此以外都是确定的：
1、所有支路都不改变输入图像的尺寸，方便最后的concatenate
2、各个支路的卷积核的kps都被固定了
3、除了pool支路，其他3个支路的输出通道都被固定了
4、不改变图像尺寸
'''
class InceptionA(nn.Module):

    def __init__(self,  in_channels, pool_features):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size = 1)#支路1*1输出通道数64

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)#支路5*5输出通道数64

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)#支路3*3输出通道数96

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)#pool支路输出通道pool_features

    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3dbl_1(x)
        branch3x3 = self.branch3x3dbl_2(branch3x3)
        branch3x3 = self.branch3x3dbl_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)#F里的命名都是小写字母，区别于nn模块
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], dim = 1)#维度是[N, C, H, W]，在C上拼接


'''
这个模块把图像大小缩小了1/2，可以看下第一个支路self.branch3x3，kps分别为3, 0, 2
假设输入长宽为x*x, 则根据kps公式，输出长宽为 (x-k+2*p)/s+1 = (x-3)/2+1
可知x必然为奇数！
实际上，这个模块只出现了一次，数据从# 35 x 35 x 288 变成了# 17 x 17 x 768
在我们的这个实现中，我们希望输入图像是可变的，而不是像原论文一样必须是299 x 299 x 3的
'''
class InceptionB(nn.Module):

    def __init__(self,  in_channels):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)#输出通道384

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)#输出通道96

    def forward(self, x):

        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)

'''
4个支路，  通道为192+channels_7x7+192+192
这里实现了把7*7的卷积核进行拆分，拆成了一个横向和竖向的 (1, 7)的长线型卷积核
不改变图像尺寸
从这里开始都是后面的模块了，按照论文的说法，从这里开始大量使用一维卷积代替二维的了
'''
class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)

'''
3个支路，  输出通道为320+192+in_channels = 512+in_channels
# 实际上D模块只出现了一次，把通道从768变成了1280
卷积核进行拆分
图像尺寸可以看branch_pool， kps为3 0 2，假设输入为x*x，输出大小 (x-3)/2+1，缩小了约2倍，x为奇数
'''
class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2) #320通道

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2) #192通道

    def forward(self, x):

        branch3x3 = self.branch3x3_1(x) 
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)

'''
4个支路，  输出通道为320+768+768+192=2048，为固定值
# 在网络中接连出现，2次
卷积核进行拆分
图像尺寸可以看branch_pool， kps为3 0 2，假设输入为x*x，输出大小 (x-3)/2+1，缩小了约2倍，x为奇数
'''
class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1) 

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0)) #384

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):

        branch1x1 = self.branch1x1(x) #320

        branch3x3 = self.branch3x3_1(x) #384
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1) #384+384=768 这里又来了个并联结构

        branch3x3dbl = self.branch3x3dbl_1(x) #448
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl) #384
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1) #384+384=768 

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)  #192

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)

'''
辅助分类器，出现在C模块之后，C模块后面接的是D模块，同时C模块的输出也送给InceptionAux来分类
在网络中，in_channels 是768，对于299*299的图像，走到这里是# 17 x 17 x 768
同时只有在训练时这个才有用，测试时不会用到这个支路
'''
class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01 #不知道干什么的，没搜到
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001 #不知道干什么的

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
