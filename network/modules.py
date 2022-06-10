# -*- coding:utf-8 -*-

"""
refer to 'https://github.com/facebookresearch/detr'
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import itertools
from collections import OrderedDict
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F


                                                                ####################################
##################################################### ********** backbone  ********  ##########################################
                                                                ####################################


class BackboneBase(nn.Module):

    def __init__(self, backbone, return_interm_layers):
        super(BackboneBase,self).__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"relu": "0","layer1": "1","layer2": "2","layer3": "3"}
        else:
            return_layers = {"layer3": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x0):
        xs = self.body(x0)
        out= {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name,return_interm_layers,dilation=False):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],pretrained=True)
        self.num_channels = 512//2 if name in ('resnet18', 'resnet34') else 2048
        super(Backbone,self).__init__(backbone, return_interm_layers)



                                                                ####################################

class BatchNorm2d_stats(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm2d_stats, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.stats = None

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
            self.stats = torch.var_mean(input,(0,2,3),unbiased=True)
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

 
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )



                                                                ####################################
##################################################### **********  segmentor  ********  ##########################################
                                                                ####################################




class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
    ):
        super().__init__()

        self.up_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('up_conv', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm2d(out_channels)),
            ('relu3', nn.ReLU(inplace=True))]))

    def forward(self, x, skip=None):
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up_block(x)
        return x    
    
    

class Decoder(nn.Module):

    def __init__(self, hidden_dim, skip, num_cls):
        super(Decoder,self).__init__()
        
        in_channels = [hidden_dim, hidden_dim//2, hidden_dim//4, hidden_dim//4]
        out_channels = [hidden_dim//2, hidden_dim//4, hidden_dim//4, hidden_dim//8]
        if skip:
            skip_channels = [0, hidden_dim//2, hidden_dim//4, hidden_dim//4]
        else:
            skip_channels = [0,0,0,0]
        
        self.up_blocks = nn.ModuleList([])
        for in_channel, out_channel, skip_channel in zip(in_channels,out_channels,skip_channels):
            
            self.up_blocks.append(UpsampleBlock(in_channel, out_channel, skip_channel))

        
        self.seg_head = nn.Conv2d(hidden_dim // 8, num_cls, kernel_size=3,padding=1)

        


    def forward(self, x, fs):
        x = self.up_blocks[0](x) 
        
        for block, f in itertools.zip_longest(self.up_blocks[1:], fs):
            #print(x.shape,f.shape)
            x = block(x,f)

        x = self.seg_head(x)
        return x
    

class UpsampleBlock1(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
    ):
        super().__init__()
        self.up_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('up_conv', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2))]))

    def forward(self, x, skip=None):
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up_block(x)
        return x    
    
    