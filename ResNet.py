#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jason Lee

@license: (C) Copyright @ Jason Lee

@contact: jiansenll@163.com

@file: ResNet.py

@time: 2019/4/14 15:39

@desc:

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # padding = 1, kernel_size = 3, the image size is not changed if stride = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # modify the input directly without allocating any additional output
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # in the second Conv stride = 1
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                # if stride is not 1, downsample
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        factor = 4  # number of shrink factor, #filters = out_channels/factor
        bottleneck_channels = out_channels // factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        block_type = config['block_type']
        assert block_type in ['basic', 'bottleneck']
        input_shape = config['input_shape']
        num_classes = config['num_classes']
        depth = config['depth']

        if block_type == 'basic':
            block = BasicBlock
            num_blocks = (depth - 2) // 6  # 6n + 2
            assert num_blocks * 6 + 2 == depth
        else:
            block = BottleneckBlock
            num_blocks = (depth - 2) // 9  # 9n + 2
            assert num_blocks * 9 + 2 == depth

        num_filters = [16, 32, 64]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[1], num_filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(block, num_filters[0], num_filters[0], num_blocks, stride=1)
        self.layer2 = self.make_layer(block, num_filters[0], num_filters[1], num_blocks, stride=2)
        self.layer3 = self.make_layer(block, num_filters[1], num_filters[2], num_blocks, stride=2)

        with torch.no_grad():
            fc_size = self.__forward__(torch.zeros(*input_shape)).view(input_shape[0], -1).shape[1] # get the shape after all Conv layers

        self.fc = nn.Linear(fc_size, num_classes)

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layer = nn.Sequential()
        for index in range(num_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                layer.add_module(block_name, block(in_channels, out_channels,
                                                   stride))  # only the first Conv will perform subsampling when stride not = 1
            else:
                layer.add_module(block_name, block(out_channels, out_channels, stride=1))

        return layer

    def __forward__(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        out = self.__forward__(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
