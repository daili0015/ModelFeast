#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-15 13:01:05
# @Last Modified by:   zcy
# @Last Modified time: 2019-03-02 17:10:28

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import sys
sys.path.append("E:/ModelFeast")
from models import densenet169
import models as model_arch

__all__ = ['DecoderRNN']


# 2D CNN encoder using ResNet-152 pretrained

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class CNNEncoder(nn.Module):
    def __init__(self, model_name = 'resnet50', img_size = (32, 32), fc_hidden1=512,\
            fc_hidden2=512, drop_p=0.3, out_channels=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # model = models.resnet152(pretrained=True)
        model = getattr(model_arch, model_name)(n_class=2, img_size=img_size)

        modules = list(model.children())[:-1]      # delete the last fc layer.
        self.kernal = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, out_channels)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        with torch.no_grad():
            for t in range(x_3d.size(1)):
                # CNNs
                x = self.kernal(x_3d[:, t, :, :, :]) # kernal model
                x = x.view(x.size(0), -1)            # flatten output of conv

                # FC layers
                x = self.bn1(self.fc1(x))
                x = F.relu(x)
                x = self.bn2(self.fc2(x))
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_p, training=self.training)
                x = self.fc3(x)

                cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


## ------------------------ DecoderRNN module ---------------------- ##
class DecoderRNN(nn.Module):
    def __init__(self, in_channels=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, \
        drop_p=0.3, n_classes=2):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = in_channels
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.n_classes = n_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.n_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of CRNN module ---------------------- ##



if __name__ == '__main__':
    a = 64
    img_size=(a, a)

    encoder = CNNEncoder(model_name = 'resnet50', img_size = img_size)
    x = torch.randn(4, 30, 3, img_size[0], img_size[1])
    # batch, depth(time), channels, h, w
    y = encoder(x)
    print(y.size())


    # decoder = DecoderRNN(n_classes=2, in_channels=56)
    # x = torch.randn(8, 30, 56)
    # # batch, depth, channels
    # y = decoder(x)

    # print(y.size())

