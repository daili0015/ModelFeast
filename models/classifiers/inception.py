#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-14 17:31:03
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 13:01:44


import logging  # 引入logging模块
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import load as TorchLoad
from models.classifiers.Inception_module import *
from base import BaseModel

__all__ = ['inception', 'inceptionv3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}
model_name = 'inception_v3_google-1a9a5a14.pth'


'''
num_classes： 输出的分类有多少种
pretrained： 是否使用预先训练模型的权重
transform_input： 是否还原归一化数据，由于预训练模型是按照0.5的均值方差归一化的，所以使用的话必须还原
aux_logits：是否使用辅助分类器
'''
def inception(n_class=1000, img_size=299, pretrained=False, pretrained_path="./pretrained/"):

    assert img_size==299 or img_size==(299, 299), "img_size must be 299 for InceptionV3!!!"
    if isinstance(img_size, (tuple, list)):
        h, w = img_size[0], img_size[1]
    else:
        h = w = img_size

    if pretrained:
        transform_input = True #使用预训练模型，必须按照模型的设定进行归一化预处理
    else:
        transform_input = False

    aux_logits = True
    # 按照预训练模型的结构 创建网络
    model = Inception3(1000, transform_input)
    model.img_size = (h, w)

    # 导入预训练模型的权值，预训练模型必须放在"./pretrained/"里
    if pretrained:
        if os.path.exists(os.path.join(pretrained_path, model_name)):
            model.load_state_dict(TorchLoad(os.path.join(pretrained_path, model_name)))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！载入权重！！")
        else:
            logging.info("pretrained 文件夹下没有，从网上下载 !!")
            model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google'], 
                model_dir = pretrained_path))
            logging.info("下载完毕！！载入权重！！")

    # 如果输出的类别不是1000，则自动调整，注意前面卷积层的权重都还在
    if n_class!=1000:
        model.fc = adaptive_classifier(model.fc.in_features, n_class)  #in_features 是2048
        init_weight(model.fc)
        print("网络的线性层自动调整为：")
        print(model.fc)
        if aux_logits:
            model.AuxLogits.fc = adaptive_classifier(model.AuxLogits.fc.in_features, n_class)  #in_features 是2048
            init_weight(model.AuxLogits.fc)
            print("辅助分类器的线性层自动调整为：")
            print(model.AuxLogits.fc)

    return model

def inceptionv3(n_class=1000, img_size=299, pretrained=False,  pretrained_path="./pretrained/"):
    return inception(n_class, img_size, pretrained, pretrained_path)


class Inception3(BaseModel):

    def __init__(self, num_classes, transform_input):
        super(Inception3, self).__init__()

        aux_logits = self.aux_logits = True
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        super(Inception3, self).isValidSize(x) #check the input size

        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits: #training是运行时model.train()时设定的
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

def init_weight(Netwrk):
    for m in Netwrk.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


'''
the structure of classifier in flexible model
features//n_class < 256 : 2 layer
features//n_class > 256 : 3 layer
'''
def adaptive_classifier(features, n_class):
    layer = []
    ratio = features//n_class
    if ratio <= 80: #专门为辅助分类器设计，辅助分类器最后features为768
        layer += [ nn.Linear(features, n_class) ]
    elif ratio <= 256:
        h1_features = int(ratio**0.5)*n_class #hidden layer's features
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True) ]
        layer += [ nn.Linear(h1_features, n_class) ]
    else:
        cube_root = int(ratio**0.33)
        h1_features = n_class*cube_root*cube_root
        h2_features = n_class*cube_root
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True), nn.Dropout()]
        layer += [ nn.Linear(h1_features, h2_features), nn.ReLU(True), nn.Dropout() ]
        layer += [ nn.Linear(h2_features, n_class) ]

    return nn.Sequential(*layer)