#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-20 17:20:40
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:52:36

import math
import torch
import logging  # 引入logging模块
import re, os
from torch import load as TorchLoad
from models.classifiers.Xception_module import Xception

__all__ = ['xception']

# 这不是pytorch官方的预训练模型，这个网址下载极慢；还是别下载了，从百度云下载下载，放在本地吧
model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}
model_names = {
    'xception':'xception-c0a72b38.pth.tar'
}

def xception(n_class, img_size=(299, 299), pretrained=False, pretrained_path="./pretrained/"):

    if isinstance(img_size, (tuple, list)):
        h, w = img_size[0], img_size[1]
    else:
        h = w = img_size
        
    model = Xception()
    model.img_size = (h, w)

    if pretrained:
        if os.path.exists(os.path.join(pretrained_path, model_names['xception'])):
            state_dict = TorchLoad(os.path.join(pretrained_path, model_names['xception']))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！直接载入！！")
            model.load_state_dict(state_dict) #权重载入完毕
        else:
            logging.info("本地文件夹下没有，请从百度云下载 !!")

    # 灵活调整
    if n_class!=1000:
        model.adaptive_fc(n_class)

    return model



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    a = 32
    img_size=(a, a)
    net = xception(10, a, True)
    y = net((torch.randn(2, 3, img_size[0], img_size[1])))
    print(y.shape)
