import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

import numpy as np
import torch

def mix_datasets_with_dirichlet(h_data1, y_data1, h_data2, y_data2, alpha, seed=49):
    """
    使用狄利克雷分布将两个数据集进行混合。
    
    参数:
    - h_data1, y_data1: 第一个数据集（均匀分布），每个元素都是tensor
    - h_data2, y_data2: 第二个数据集（完全异质性分布），每个元素都是tensor
    - alpha: 狄利克雷分布的参数
    - seed: 随机种子，确保可复现性

    返回:
    - h_data_mixed, y_data_mixed: 混合后的数据集，list中的每个元素都是tensor
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    num_nodes = len(h_data1)
    h_data_mixed = []
    y_data_mixed = []
    
    for i in range(num_nodes):
        # 生成狄利克雷分布的权重
        dirichlet_weights = np.random.dirichlet([alpha, alpha])

        # 计算从每个数据集中提取的样本数量
        len1 = int(dirichlet_weights[0] * len(h_data1[i]))
        len2 = int(dirichlet_weights[1] * len(h_data2[i]))

        # 混合数据集
        h_data_mixed_i = torch.cat([
            h_data1[i][:len1],
            h_data2[i][:len2]
        ], dim=0)
        
        y_data_mixed_i = torch.cat([
            y_data1[i][:len1],
            y_data2[i][:len2]
        ], dim=0)
        
        # 打乱混合后的数据
        perm = torch.randperm(len(h_data_mixed_i))
        h_data_mixed.append(h_data_mixed_i[perm])
        y_data_mixed.append(y_data_mixed_i[perm])
    
    return h_data_mixed, y_data_mixed



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

def prepare_node_5_hard_shuffled():
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 转换回PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 打乱训练数据
    perm = torch.randperm(X_train.size(0))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # 将数据分为5份
    chunk_size = len(X_train) // 5
    h_data = [X_train[i * chunk_size:(i + 1) * chunk_size] for i in range(5)]
    y_data = [y_train[i * chunk_size:(i + 1) * chunk_size] for i in range(5)]
    
    # 如果有剩余数据，将其分配给最后一个节点
    if len(X_train) % 5 != 0:
        h_data[-1] = torch.cat((h_data[-1], X_train[5 * chunk_size:]), dim=0)
        y_data[-1] = torch.cat((y_data[-1], y_train[5 * chunk_size:]), dim=0)
    
    return h_data, y_data, X_test, y_test

# 获取node=10的分类数据
def prepare_node_10_hard_shuffled():
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 转换回PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    X_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 X_train
    y_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 y_train
    
    # 打乱训练数据
    perm = torch.randperm(X_train.size(0))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # 将数据分为10份
    chunk_size = len(X_train) // 10
    X_train_list = [X_train[i * chunk_size:(i + 1) * chunk_size] for i in range(10)]
    y_train_list = [y_train[i * chunk_size:(i + 1) * chunk_size] for i in range(10)]
    
    # 如果有剩余数据，将其分配给最后一个节点
    if len(X_train) % 10 != 0:
        X_train_list[-1] = torch.cat((X_train_list[-1], X_train[10 * chunk_size:]), dim=0)
        y_train_list[-1] = torch.cat((y_train_list[-1], y_train[10 * chunk_size:]), dim=0)
    
    return X_train_list, y_train_list, X_test, y_test

def prepare_node_5_hard_mix(alpha=1,seed=42):
    """ alpha ——> 0, 高异质性; alpha ——> infty, 均匀分布 """
    h_data1, y_data1, X_test1, y_test1 = prepare_node_5_hard()#大异质性
    h_data2, y_data2, X_test2, y_test2 = prepare_node_5_hard_shuffled()#均匀
    h_data_mixed, y_data_mixed = mix_datasets_with_dirichlet(h_data1=h_data1, y_data1=y_data1, h_data2=h_data2, y_data2=y_data2, alpha=alpha,seed=seed)
    X_test_mixed = torch.cat((X_test1, X_test2), dim=0)
    y_test_mixed = torch.cat((y_test1, y_test2), dim=0)
    return h_data_mixed,y_data_mixed,X_test_mixed,y_test_mixed

def prepare_node_10_hard_mix(alpha=1,seed=42):
    """ alpha ——> 0, 高异质性; alpha ——> infty, 均匀分布 """
    h_data1, y_data1, X_test1, y_test1 = prepare_node_10_hard()#大异质性
    h_data2, y_data2, X_test2, y_test2 = prepare_node_10_hard_shuffled()#均匀
    h_data_mixed, y_data_mixed = mix_datasets_with_dirichlet(h_data1=h_data1, y_data1=y_data1, h_data2=h_data2, y_data2=y_data2, alpha=alpha,seed=seed)
    X_test_mixed = torch.cat((X_test1, X_test2), dim=0)
    y_test_mixed = torch.cat((y_test1, y_test2), dim=0)
    return h_data_mixed,y_data_mixed,X_test_mixed,y_test_mixed