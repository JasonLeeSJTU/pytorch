import torch
import torchvision
import torchvision.transforms as ts
import matplotlib.pyplot as plt
import numpy as np


# normalize PILImages to Tensors of range [-1, 1]
transform = ts.Compose([
    ts.ToTensor(),
    ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
        num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
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
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images, padding=2))
    print(' '.join('%s' % classes[labels[j]] for j in range(4)))
    print(' '.join(classes[labels[j]] for j in range(4)))
