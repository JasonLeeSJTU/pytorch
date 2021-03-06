#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jason Lee

@license: (C) Copyright @ Jason Lee

@contact: jiansenll@163.com

@file: cnn.py

@time: 2019/4/7 21:43

@desc:

'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.bn = bn

    def forward(self, x):
        if not self.bn:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.bn1(self.conv1(x))
            x = self.pool(F.relu(x))
            x = self.bn2(self.conv2(x))
            x = self.pool(F.relu(x))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))
            x = self.fc3(x)

        return x