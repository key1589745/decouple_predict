# -*- coding:utf-8 -*-

import math, time

from itertools import chain

import torch

import torch.nn.functional as F

from torch import nn

from .modules import *

import torchvision.transforms as transforms




class ResUnet(nn.Module):

    def __init__(self, configs, num_cls):
        super(ResUnet,self).__init__()
        
        self.variance = False
        if configs.mode == 'alea':
            self.decoder_v = Decoder(configs.hidden_dim,configs.skip,num_cls=1)
            self.variance = True
            
        self.decoder = Decoder(configs.hidden_dim,configs.skip,num_cls=num_cls)    
        self.backbone = Backbone(configs.name,configs.skip)


    def forward(self, x):
        
        x = x.repeat(1,3,1,1)
        
        x = list(self.backbone(x).values())[::-1]     

        pred = self.decoder(x[0], x[1:])
        
        if self.variance and self.training:
            var = self.decoder_v(x[0], x[1:])
            return pred, var
        else:
            return pred