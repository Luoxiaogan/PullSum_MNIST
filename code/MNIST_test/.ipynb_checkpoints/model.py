import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 64)  # 第一层将输入从784维度减少到64维度
        self.fc2 = nn.Linear(64, 1)    # 第二层将64维度减少到1维度
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))   # 第一层之后使用ReLU激活函数
        x = torch.sigmoid(self.fc2(x)) # 第二层之后使用Sigmoid激活函数
        return x



""" class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x """