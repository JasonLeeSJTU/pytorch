#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jason Lee

@license: (C) Copyright @ Jason Lee

@contact: jiansenll@163.com

@file: ResNet_main.py

@time: 2019/4/14 14:54

@desc:

'''
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ResNet import ResNet

# use GPU if available
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
epochs = 240
batch_size = 128
lr = 0.1
momentum = 0.9

# prepare data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # padding 4 pixels around the image, then crop 32x32 patches randomly
    transforms.RandomHorizontalFlip(0.5),   # horizontally flip half the image
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model
config = {
    'block_type': 'basic',
    'depth': 20,
    'input_shape': (batch_size, 3, 32, 32),
    'num_classes': 10
}
net = ResNet(config)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)    # with L2 penalty


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train
if __name__ == '__main__':
    with open("acc.txt", "w") as acc:
        with open("log.txt", "w") as log:
            for epoch in range(epochs):
                print(f"Epoch: {epoch + 1}")
                # adjust learning rate
                if epoch < 100:
                    new_lr = lr
                    adjust_learning_rate(optimizer, new_lr)
                elif epoch < 180:
                    new_lr = lr*0.1
                    adjust_learning_rate(optimizer, new_lr)
                else:
                    new_lr = lr*0.01
                    adjust_learning_rate(optimizer, new_lr)

                net.train()

                sum_loss = 0.0
                total = 0.0
                correct = 0.0
                for i, data in enumerate(train_loader, 0):  # i is the number of batches
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    print(f"[epoch: {epoch+1}, iter: {i+1 + epoch*length}] Loss: {sum_loss/(i+1)}, Acc: {correct*100/total}")
                    log.write(f"{epoch+1} {i+1+epoch*length} | Loss: {sum_loss/(i+1)} | Acc: {correct*100/total}\n")
                    log.flush()

                # test
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        print(f"Testing Acc: {100*correct/total}")
                        acc.write(f"epoch={epoch+1}, Acc={100*correct/total}\n")
                        acc.flush()

            print("Training finished.")

            # accuracy
            correct = [0] * 10
            total = [0] * 10
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    c = predicted == labels
                    for i in range(4):
                        label = labels[i]
                        total[label] += 1
                        correct[label] += c[i].item()

            for i in range(10):
                print(f'Accuracy of {classes[i]}: {100 * correct[i] / total[i]} %')

