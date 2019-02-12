#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-01-19 10:04:27
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-11 11:51:49

import logging  # 引入logging模块
import re, os
import torch
from torch import load as TorchLoad
import torch.utils.model_zoo as model_zoo

from models.classifiers.DenseNet_module import DenseNet

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

model_names = {
    'densenet121': 'densenet121-a639ec97.pth',
    'densenet169': 'densenet169-b2777c0a.pth',
    'densenet201': 'densenet201-c1103571.pth',
    'densenet161': 'densenet161-8d451a50.pth',
}


# 图像大小img_size没啥用处，为了跟别的网络接口一样，故而这么设定
def densenet121(n_class, img_size = (32, 32), pretrained=False, pretrained_path="./pretrained/"):
    Net_param = {'num_init_features':64, 'growth_rate':32, 'block_config':(6, 12, 24, 16),
                'url':model_urls['densenet121'], 'model_name':model_names['densenet121'],
                'img_size':img_size }
    return get_densenet(Net_param, n_class, pretrained, pretrained_path)

def densenet161(n_class, img_size = (32, 32), pretrained=False, pretrained_path="./pretrained/"):
    Net_param = {'num_init_features':96, 'growth_rate':48, 'block_config':(6, 12, 36, 24),
                'url':model_urls['densenet161'], 'model_name':model_names['densenet161'],
                'img_size':img_size }
    return get_densenet(Net_param, n_class, pretrained, pretrained_path) 

def densenet169(n_class, img_size = (32, 32), pretrained=False, pretrained_path="./pretrained/"):
    Net_param = {'num_init_features':64, 'growth_rate':32, 'block_config':(6, 12, 32, 32),
                'url':model_urls['densenet169'], 'model_name':model_names['densenet169'],
                'img_size':img_size }
    return get_densenet(Net_param, n_class, pretrained, pretrained_path)

def densenet201(n_class, img_size = (32, 32), pretrained=False, pretrained_path="./pretrained/"):
    Net_param = {'num_init_features':64, 'growth_rate':32, 'block_config':(6, 12, 48, 32),
                'url':model_urls['densenet201'], 'model_name':model_names['densenet201'],
                'img_size':img_size }
    return get_densenet(Net_param, n_class, pretrained, pretrained_path)


def get_densenet(Net_param, n_class, pretrained=False, pretrained_path="./pretrained/"):
    '''
    Net_param：网络参数，只与网络类型有关 
                包含 模型url 模型文件名字 growth_rate block_config
    n_class：输出类别
    pretrained：是否使用预训练模型
    img_size: img_size
    '''

    if isinstance(Net_param['img_size'], (tuple, list)):
        h, w = Net_param['img_size'][0], Net_param['img_size'][1]
    else:
        h = w = Net_param['img_size']    

    model = DenseNet(num_init_features=Net_param['num_init_features'], 
        growth_rate=Net_param['growth_rate'], block_config=Net_param['block_config'])
    model.img_size = (h, w)

    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        
        if os.path.exists(os.path.join(pretrained_path, Net_param['model_name'])):
            state_dict = TorchLoad(os.path.join("./pretrained", Net_param['model_name']))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！直接载入！！")
        else:
            logging.info("pretrained 文件夹下没有，从网上下载 !!")
            state_dict = model_zoo.load_url(Net_param['url'], model_dir = pretrained_path)
            logging.info("下载完毕！！载入权重！！")

        # 导入进来
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict) #权重载入完毕

    # 灵活调整
    if n_class!=1000:
        model.adaptive_set_fc(n_class)

    return model

if __name__ == '__main__':
    a = 64
    img_size=(a, a)
    net = densenet201(10, False)
    y = net((torch.randn(2, 3, img_size[0], img_size[1])))
    print(y.size())

