import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as ts
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from cnn import Net


# normalize PILImages to Tensors of range [-1, 1]
transform = ts.Compose([
    ts.ToTensor(),
    ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
        num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
        num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck')

# function to show an image
def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


if __name__ == '__main__':
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images, padding=2))
    # print(' '.join(classes[labels[j]] for j in range(4)))

    bn = False
    net = Net(bn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.to(device)

    # training the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            #optimize
            optimizer.step()

            # print
            running_loss += loss.item()
            # if i % 2000 == 1999:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss / (i+1)}')
                # running_loss = 0.0
    print('Finished training.')

    # test the network
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    # imshow(torchvision.utils.make_grid(images.cpu()))
    print('Ground Truth: ', ' '.join(classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(classes[predicted[j]] for j in range(4)))

    # accuracy
    correct = [0] * 10
    total = [0] * 10
    with torch.no_grad():
        for data in testloader:
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

