import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        # 如果输入输出通道数不同，或 stride 不为 1，则通过 1x1 卷积调整尺寸
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet1(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet1, self).__init__()
        self.layer1 = BasicBlock1(3, 16, stride=1)
        self.layer2 = BasicBlock1(16, 32, stride=2)
        self.layer3 = BasicBlock1(32, 64, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

class SimpleResNet2(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet2, self).__init__()
        self.layer1 = BasicBlock2(3, 16, stride=1)
        self.layer2 = BasicBlock2(16, 32, stride=2)
        self.layer3 = BasicBlock2(32, 64, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

