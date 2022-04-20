import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from src.layers.full_linked_conv import FullLinkedConv
import numpy as np

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class FullTensorizedBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, ranks, stride=1, option='A'):
        super(FullTensorizedBasicBlock, self).__init__()
        in_size = np.prod(in_planes)
        size = np.prod(planes)
        self.conv1 = FullLinkedConv(in_planes, planes, ranks=ranks, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(size)
        self.conv2 = FullLinkedConv(planes, planes, ranks=ranks, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_size != size:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, size//4, size//4), "constant", 0))
            elif option == 'B':
                assert False

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FullTensorizedResNet(nn.Module):
    def __init__(self, block, num_blocks, ranks, num_classes=10):
        super(FullTensorizedResNet, self).__init__()
        self.in_planes = (4, 2, 2)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, (4, 2, 2), ranks, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, (4, 4, 2), ranks, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, (4, 4, 4), ranks, num_blocks[2], stride=2)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, ranks, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, ranks, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.classifier(out)
        return out


def tensorized_resnet32(ranks):
    return FullTensorizedResNet(FullTensorizedBasicBlock, [5, 5, 5], ranks)

def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()