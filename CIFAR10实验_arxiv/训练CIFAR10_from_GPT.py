import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ===========================
# 设置设备（使用CPU）
# ===========================
device = torch.device('cpu')

# ===========================
# 超参数设置（可修改学习率、训练轮数和权重衰减）
# ===========================
num_epochs = 10          # 训练轮数，可根据需要修改
learning_rate = 0.001    # 学习率，可根据需要修改
weight_decay = 0         # 权重衰减（L2正则化），可根据需要修改

# ===========================
# 数据集路径（请填入您自己的路径）
# ===========================
data_path = 'path_to_cifar10_dataset'  # 替换为您的CIFAR10数据集路径

# ===========================
# 数据预处理和加载
# ===========================
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # 标准化
                         (0.5, 0.5, 0.5))
])

# 加载训练集
train_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                             train=True,
                                             transform=transform,
                                             download=False)

# 加载测试集
test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=False,
                                            transform=transform,
                                            download=False)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# ===========================
# 定义卷积神经网络模型
# ===========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积，输入通道3，输出通道16，卷积核大小5，填充2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        # 最大池化层，池化窗口2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二层卷积
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        # 全连接层1
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)
        # 输出层
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # 展平
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ===========================
# 初始化模型、损失函数和优化器
# ===========================
model = CNN().to(device)  # 将模型移动到CPU

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam优化器

# ===========================
# 定义保存训练和测试指标的列表
# ===========================
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# ===========================
# 训练模型
# ===========================
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    # 遍历训练数据
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # 将数据移动到CPU
        labels = labels.to(device)

        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)  # 累计损失
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 统计正确预测的样本数

    # 计算平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # ===========================
    # 在测试集上评估模型
    # ===========================
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # 计算测试集的平均损失和准确率
    test_epoch_loss = test_loss / test_total
    test_epoch_acc = 100 * test_correct / test_total
    test_losses.append(test_epoch_loss)
    test_accuracies.append(test_epoch_acc)

    # 输出本轮训练和测试的损失和准确率
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
          f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.2f}%')

# ===========================
# 绘制损失曲线
# ===========================
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# ===========================
# 绘制准确率曲线
# ===========================
plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curves')
plt.legend()
plt.show()
