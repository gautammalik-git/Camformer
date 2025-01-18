import collections
from collections import OrderedDict
from itertools import repeat
import json
import random

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def convolution_output_size(input_size, kernel, padding, stride):
    """
    Calculate the output size of a convolutional layer.
    """
    out_height = int(((input_size[0] + 2*padding[0] - (kernel[0] - 1) - 1)/stride[0])+1)
    out_width = int(((input_size[1] + 2*padding[1] - (kernel[1] - 1) - 1)/stride[1])+1)
    
    return (out_height, out_width)


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def conv_block(in_f, out_f, kernel, pool_kernel, stride, pool_stride, padding, dilation=1, dropout=0.1):
    """
    Return a 2D convolutional block.
    """
    modules = []
    if padding == "same":
        modules.append(Conv2dSame(in_channels=in_f, out_channels=out_f, kernel_size=kernel, stride=stride, dilation=dilation))
    else:
        modules.append(nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation))
    modules.append(nn.BatchNorm2d(out_f))
    modules.append(nn.ReLU())
    if pool_kernel != (1,1):
        modules.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
    modules.append(nn.Dropout(dropout))

    layer = nn.Sequential(*modules)
    return layer


def lin_block(in_f, out_f, dropout=0.0):
    """
    Return a linear block.
    """
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

# From https://github.com/pytorch/pytorch/issues/3867#issuecomment-974159134
class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0, 
    so we need to implement the internal torch logic manually. 

    Currently the ``RuntimeError`` is
    
    "PyTorch convert function for op '_convolution_mode' not implemented"
    """

    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)
        _pair = _ntuple(2)
        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_, 
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class CNN(nn.Module):   
    
    """
    Define the neural network structure.
    """
    
    def __init__(self, feature_height, feature_width, batch_size, 
                 print_size=False,
                 input_channels=4,
                 out_channels=[256, 256, 256], 
                 kernels=[(12,1),(4,1),(4,1)], 
                 pool_kernels=[(1,1),(1,1),(1,1)], 
                 paddings=[(0,0), (0,0), (0,0)], 
                 strides=[(1,1), (1,1), (1,1)], 
                 pool_strides=[(1,1), (1,1), (1,1)], 
                 dropouts=[0.2, 0.2, 0.2], 
                 linear_output=[64], 
                 linear_dropouts=[0.0]):
        
        super(CNN, self).__init__()

        if print_size:
            print(locals())
        
        out_size = (feature_height, feature_width) # Set a start size
        
        if print_size:
            print("Sizes: ", end="")
        # Calculate layer output sizes, output if print_size = True
        if paddings == "same":
            paddings = ["same" for x in range(len(out_channels))]
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")
        else:
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, kernels[n], paddings[n], strides[n])
                print(out_size[0],"x",out_size[1],end="\t")
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")

        # Calculate the linear input size.
        output_size = int(out_channels[-1] * out_size[0] * out_size[1]) # Input to the linear layers
        if print_size:
            print("Linear input size: %s" % output_size)
      
        # Define the convolutional layers
        self.CNN_layers = nn.ModuleList()
        conv_sizes = [x for x in zip([input_channels] + out_channels, out_channels)]
        for n, sizes in enumerate(conv_sizes):
            conv_blocks = conv_block(sizes[0], sizes[1], 
                                     kernel=kernels[n], 
                                     pool_kernel=pool_kernels[n], 
                                     stride=strides[n], 
                                     pool_stride=pool_strides[n], 
                                     padding="same", 
                                     dilation=1, 
                                     dropout=dropouts[n])
            self.CNN_layers.append(nn.Sequential(*conv_blocks))
               
        # Define the fully connected layers
        if len(linear_output) == 0:
            linear_blocks = nn.Sequential(nn.Linear(output_size, 1))
        else:
            linear_sizes = [x for x in zip([output_size] + linear_output, linear_output + [1])]
            linear_blocks = [lin_block(in_f, out_f, dropout=linear_dropouts[n]) for n, (in_f, out_f) in enumerate(linear_sizes[:-1])]
            last_layer = nn.Sequential(nn.Linear(linear_sizes[-1][0], linear_sizes[-1][1]))
            linear_blocks.append(last_layer)
        self.linear_layers = nn.Sequential(*linear_blocks)
        
    
    def forward(self, x):
        for i in range(0, len(self.CNN_layers), 2):
            x = self.CNN_layers[i](x)
            ShortCut = x
            x = self.CNN_layers[i+1](x)
            x = x + ShortCut
       
        # FC layers
        x = x.view(x.size(0), -1) # flatten before FC layers
        x = self.linear_layers(x)
        x = torch.flatten(x)
        
        return x
    

def initialize_weights_he(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
