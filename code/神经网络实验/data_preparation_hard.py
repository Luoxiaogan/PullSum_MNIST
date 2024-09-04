import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST data
def load_mnist_data():
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    X_train = trainset.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_train = trainset.targets.numpy().astype(np.int64)
    X_test = testset.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_test = testset.targets.numpy().astype(np.int64)
    
    return X_train, X_test, y_train, y_test

# 获取node=5的分类数据
def prepare_node_5_hard():
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 使用train_test_split进一步分割训练集和测试集（可选）
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.0, random_state=42)

    # 转换回PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    h_data = [[] for _ in range(5)]
    y_data = [[] for _ in range(5)]
    
    for i in range(len(y_train)):
        label = y_train[i].item()
        group_index = label // 2  # 根据标签分组，0-1 -> 0, 2-3 -> 1, 4-5 -> 2, 6-7 -> 3, 8-9 -> 4
        h_data[group_index].append(X_train[i])
        y_data[group_index].append(y_train[i])
    
    h_data = [torch.stack(group_data) for group_data in h_data]
    y_data = [torch.tensor(group_labels, dtype=torch.long) for group_labels in y_data]
    
    return h_data, y_data, X_test, y_test

# 获取node=10的分类数据
def prepare_node_10_hard():
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 使用train_test_split进一步分割训练集和测试集（可选）
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.0, random_state=42)

    # 转换回PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    X_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 X_train
    y_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 y_train
    
    for i in range(len(y_train)):
        label = y_train[i].item()
        X_train_classified[label].append(X_train[i])
        y_train_classified[label].append(y_train[i])
    
    X_train_list = [torch.stack(class_data) for class_data in X_train_classified]
    y_train_list = [torch.tensor(class_labels, dtype=torch.long) for class_labels in y_train_classified]
    
    return X_train_list, y_train_list, X_test, y_test
