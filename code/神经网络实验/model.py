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
    
class MNISTClassifier_2layer_1(nn.Module):
    def __init__(self):
        super(MNISTClassifier_2layer_1, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 第一层，输入 784 维度，输出 128 维度
        self.fc2 = nn.Linear(128, 10)   # 第二层，输入 128 维度，输出 10 维度（对应 10 个类别）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层使用 ReLU 激活函数
        x = self.fc2(x)          # 输出层不使用激活函数，因为后面会使用 CrossEntropyLoss，它内部包含 softmax
        return x
    
class MNISTClassifier_2layer_2(nn.Module):
    def __init__(self,p=0.5):
        super(MNISTClassifier_2layer_2, self).__init__()
        self.fc1 = nn.Linear(784, 128)          # 第一层，全连接层
        self.bn1 = nn.BatchNorm1d(128)          # 第一层，批归一化
        self.fc2 = nn.Linear(128, 10)           # 第二层，全连接层
        self.dropout = nn.Dropout(p=p)          # Dropout 层，防止过拟合
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))       # 第一层，先全连接，再批归一化，最后 ReLU 激活
        x = self.dropout(x)                     # Dropout 层，随机丢弃部分神经元
        x = self.fc2(x)                         # 第二层，全连接输出层（不使用激活函数）
        return x
    
class MNISTClassifier_3layer(nn.Module):
    def __init__(self):
        super(MNISTClassifier_3layer, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 第一层将输入从784维度减少到128维度
        self.fc2 = nn.Linear(128, 64)   # 第二层将128维度减少到64维度
        self.fc3 = nn.Linear(64, 10)    # 第三层输出10维度，对应0-9的十个类别
    
    def forward(self, x):
        x = F.relu(self.fc1(x))         # 第一层之后使用ReLU激活函数
        x = F.relu(self.fc2(x))         # 第二层之后使用ReLU激活函数
        x = self.fc3(x)                 # 第三层的输出用于分类，因此不使用激活函数
        return x
    
class MNISTClassifier_4layer_1(nn.Module):
    def __init__(self):
        super(MNISTClassifier_4layer_1, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class MNISTClassifier_4layer_2(nn.Module):
    def __init__(self,p=0.5):
        super(MNISTClassifier_4layer_2, self).__init__()
        self.fc1 = nn.Linear(784, 256)           # 增加第一层神经元数量
        self.bn1 = nn.BatchNorm1d(256)           # 批归一化
        self.fc2 = nn.Linear(256, 128)           # 增加第二层
        self.bn2 = nn.BatchNorm1d(128)           # 批归一化
        self.fc3 = nn.Linear(128, 64)            # 增加第三层
        self.bn3 = nn.BatchNorm1d(64)            # 批归一化
        self.fc4 = nn.Linear(64, 10)             # 输出层
        self.dropout = nn.Dropout(p=p)           # Dropout 层
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))        # 第一层
        x = self.dropout(x)                      # Dropout
        x = F.relu(self.bn2(self.fc2(x)))        # 第二层
        x = self.dropout(x)                      # Dropout
        x = F.relu(self.bn3(self.fc3(x)))        # 第三层
        x = self.dropout(x)                      # Dropout
        x = self.fc4(x)                          # 输出层
        return x