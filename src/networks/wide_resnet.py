import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial


class Wide_Block(nn.Module):
    def __init__(self, in_maps, maps, stride):
        super(Wide_Block, self).__init__()
        self.in_maps = in_maps
        self.maps = maps
        self.bn1 = nn.BatchNorm2d(in_maps)
        self.conv1 = nn.Conv2d(in_maps, maps, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(maps)
        self.conv2 = nn.Conv2d(maps, maps, 3, stride=1, padding=1)
        if self.in_maps != self.maps:
            self.convdim = nn.Conv2d(in_maps, maps, 1, stride=stride)
    
    def forward(self, x):
        o1 = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(o1)
        o2 = F.relu(self.bn2(y), inplace=True)
        z = self.conv2(o2)
        if self.in_maps != self.maps:
            return z + self.convdim(o1)
        else:
            return z + x


class Wide_ResNet(nn.Module):
    def __init__(self, depth, width, dropout, num_classes):
        super(Wide_ResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [16] + [int(v * width) for v in (16, 32, 64)]

        def group(stride, in_maps, maps):
            return nn.Sequential(*([Wide_Block(in_maps, maps, stride)] + [Wide_Block(maps, maps, 1) for i in range(n - 1)]))

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.group1 = group(1, widths[0], widths[1])
        self.group2 = group(2, widths[1], widths[2])
        self.group3 = group(2, widths[2], widths[3])
        self.bn = nn.BatchNorm2d(widths[3])
        self.classifier = nn.Sequential(
            nn.AvgPool2d(8, 1, 0),
            nn.Flatten(),
            nn.Linear(widths[3], num_classes)
        )

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.running_mean, 0)
                torch.nn.init.constant_(m.running_var, 1)
        
        self.apply(init_weights)

    def forward(self, input):
        x = self.conv1(input)
        g0 = self.group1(x)
        g1 = self.group2(g0)
        g2 = self.group3(g1)
        o = F.relu(self.bn(g2))
        o = self.classifier(o)
        return o
