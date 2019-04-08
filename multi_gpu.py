#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jason Lee

@license: (C) Copyright @ Jason Lee

@contact: jiansenll@163.com

@file: multi_gpu.py

@time: 2019/4/8 21:09

@desc:

'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print(f'\tIn Model: input size {input.size()}, output size {output.size()}')
        return output



input_size = 5
output_size = 2
batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

with torch.no_grad():
    for data in rand_loader:
        inputs = data.to(device)
        outputs = model(inputs)
        print(f"Outside: input size {inputs.size()} output size {outputs.size()}")