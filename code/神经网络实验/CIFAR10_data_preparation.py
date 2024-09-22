import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
import tarfile
import imgaug.augmenters as iaa
import torchfile

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

""" def load_cifar10_data():
    # 设置路径
    cifar10_dir = '/home/bluefog/GanLuo/PullSum_MNIST/code/神经网络实验/data_cifar10'
    
    # 加载所有训练批次
    X_train = []
    y_train = []
    for i in range(1, 6):  # 批次1到5
        batch_path = os.path.join(cifar10_dir, f'data_batch_{i}')
        X_batch, y_batch = load_cifar10_batch(batch_path)
        X_train.append(X_batch)
        y_train.append(y_batch)
    
    # 合并所有训练批次
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    # 加载测试集
    X_test, y_test = load_cifar10_batch(os.path.join(cifar10_dir, 'test_batch'))
    
    # 正则化数据，与 `torchvision` 中的标准化保持一致
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    
    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std
    
    return X_train, X_test, y_train, y_test """

def load_cifar10_batch(batch_path):
    """ 加载单个批次的数据 """
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        # 读取数据和标签
        X = batch[b'data']
        y = batch[b'labels']
        # CIFAR-10 中每个图像为 3 x 32 x 32 的彩色图像，需要重新reshape
        X = X.reshape(-1, 3, 32, 32).astype('float32')
        return X, np.array(y)
    
def load_cifar10_from_t7(t7_path):
    dataset = torchfile.load(t7_path)

    # 获取训练和测试数据
    X_train = torch.from_numpy(dataset[b'trainData'][b'data'])  # 转换为张量
    #X_train = X_train.permute(0, 2, 3, 1).numpy()  # 转换为 (N, H, W, C)
    y_train = dataset[b'trainData'][b'labels'] - 1  # 训练标签，形状应为 (50000,)
    
    # 获取测试数据
    X_test = torch.from_numpy(dataset[b'testData'][b'data'])  # 转换为张量
    #X_test = X_test.permute(0, 2, 3, 1).numpy()  # 转换为 (N, H, W, C)
    y_test = dataset[b'testData'][b'labels'] - 1  # 测试标签，形状应为 (10000,)

    return X_train, y_train, X_test, y_test

def augment_data(X):
    """ 使用 imgaug 进行数据增强 """
    from imgaug import augmenters as iaa

    # 定义数据增强的流水线
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转 50% 的图像
        iaa.Crop(percent=(0, 0.1)),  # 随机裁剪图像边缘
        iaa.Affine(rotate=(-10, 10)),  # 随机旋转 -10 到 10 度
        iaa.Multiply((0.8, 1.2)),  # 随机调整亮度
        iaa.LinearContrast((0.75, 1.5))  # 随机调整对比度
    ])

    # 转换数据维度为 (batch_size, 32, 32, 3) 以适应 imgaug
    X = np.transpose(X, (0, 2, 3, 1))
    X_augmented = seq(images=X)
    # 转换回 (batch_size, 3, 32, 32)
    X_augmented = np.transpose(X_augmented, (0, 3, 1, 2))

    return X_augmented

def load_cifar10_data(t7_path=None, augment=False, use_t7_data=False):
    """ 加载 CIFAR-10 数据，augment 参数决定是否进行数据增强，use_t7_data 参数决定数据源 """
    
    if use_t7_data:
        # 从 t7 文件加载数据
        X_train, y_train, X_test, y_test=load_cifar10_from_t7(t7_path=t7_path)
        return X_train, X_test, y_train, y_test
    else:
        # 设置路径
        cifar10_dir = '/root/GanLuo/PullSum_MNIST/code/神经网络实验/CIFAR10data/cifar-10-batches-py'

        # 加载所有训练批次
        X_train = []
        y_train = []
        for i in range(1, 6):  # 批次1到5
            batch_path = os.path.join(cifar10_dir, f'data_batch_{i}')
            X_batch, y_batch = load_cifar10_batch(batch_path)
            X_train.append(X_batch)
            y_train.append(y_batch)

        # 合并所有训练批次
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # 加载测试集
        test_batch_path = os.path.join(cifar10_dir, 'test_batch')
        X_test, y_test = load_cifar10_batch(test_batch_path)

    # 如果 augment=True，则对训练集进行数据增强
    if augment:
        X_train = augment_data(X_train)

    # 正则化数据，与 `torchvision` 中的标准化保持一致
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std

    return X_train, X_test, y_train, y_test

# 获取node=5的分类数据
def cifar10_prepare_node_5_hard(augment=False):
    X_train, X_test, y_train, y_test = load_cifar10_data(augment=augment)

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
def cifar10_prepare_node_10_hard(augment=False):
    X_train, X_test, y_train, y_test = load_cifar10_data(augment=augment)

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

def cifar10_prepare_node_5_hard_shuffled(augment=False):
    X_train, X_test, y_train, y_test = load_cifar10_data(augment=augment)

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
def cifar10_prepare_node_10_hard_shuffled(augment=False):
    X_train, X_test, y_train, y_test = load_cifar10_data(augment=augment)

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

def cifar10_prepare_node_5_hard_mix(alpha=1,seed=42,augment=False):
    """ alpha ——> 0, 高异质性; alpha ——> infty, 均匀分布 """
    h_data1, y_data1, X_test1, y_test1 = cifar10_prepare_node_5_hard(augment=augment)#大异质性
    h_data2, y_data2, X_test2, y_test2 = cifar10_prepare_node_5_hard_shuffled(augment=augment)#均匀
    h_data_mixed, y_data_mixed = mix_datasets_with_dirichlet(h_data1=h_data1, y_data1=y_data1, h_data2=h_data2, y_data2=y_data2, alpha=alpha,seed=seed)
    X_test_mixed = torch.cat((X_test1, X_test2), dim=0)
    y_test_mixed = torch.cat((y_test1, y_test2), dim=0)
    return h_data_mixed,y_data_mixed,X_test_mixed,y_test_mixed

def cifar10_prepare_node_5_hard_linear_mix(p1=0.5, seed=42,augment=False):
    """ 
    p1 是高异质性数据抽样的比例，p2=1-p1 是均匀分布数据抽样的比例
    """
    # 获取原始数据
    h_data1, y_data1, X_test1, y_test1 = cifar10_prepare_node_5_hard(augment=augment)  # 大异质性，对应 p1
    h_data2, y_data2, X_test2, y_test2 = cifar10_prepare_node_5_hard_shuffled(augment=augment)  # 均匀，对应 p2
    
    p2 = 1 - p1
    
    # 设置随机种子
    np.random.seed(seed)

    h_data_mixed = []
    y_data_mixed = []

    # 依次对每个节点进行抽样
    for h1, y1, h2, y2 in zip(h_data1, y_data1, h_data2, y_data2):
        # 确定每个数据集的抽样数量
        len_h1 = len(h1)
        len_h2 = len(h2)
        
        # 计算抽样数量（不能整除时向下取整）
        sample_size_h1 = int(len_h1 * p1)
        sample_size_h2 = int(len_h2 * p2)
        
        # 从 h1, y1 和 h2, y2 中分别抽样
        indices_h1 = np.random.choice(len_h1, sample_size_h1, replace=False)
        indices_h2 = np.random.choice(len_h2, sample_size_h2, replace=False)
        
        h1_sampled = h1[indices_h1]
        y1_sampled = y1[indices_h1]
        h2_sampled = h2[indices_h2]
        y2_sampled = y2[indices_h2]

        # 将两个抽样结果拼接
        h_mixed = torch.cat([h1_sampled, h2_sampled], dim=0)
        y_mixed = torch.cat([y1_sampled, y2_sampled], dim=0)

        h_data_mixed.append(h_mixed)
        y_data_mixed.append(y_mixed)
    
    # 测试数据直接拼接
    X_test = torch.cat((X_test1, X_test2), dim=0)
    y_test = torch.cat((y_test1, y_test2), dim=0)
    
    return h_data_mixed, y_data_mixed, X_test, y_test

def cifar10_prepare_node_10_hard_mix(alpha=1,seed=42,augment=False):
    """ alpha ——> 0, 高异质性; alpha ——> infty, 均匀分布 """
    h_data1, y_data1, X_test1, y_test1 = cifar10_prepare_node_10_hard(augment=augment)#大异质性
    h_data2, y_data2, X_test2, y_test2 = cifar10_prepare_node_10_hard_shuffled(augment=augment)#均匀
    h_data_mixed, y_data_mixed = mix_datasets_with_dirichlet(h_data1=h_data1, y_data1=y_data1, h_data2=h_data2, y_data2=y_data2, alpha=alpha,seed=seed)
    X_test_mixed = torch.cat((X_test1, X_test2), dim=0)
    y_test_mixed = torch.cat((y_test1, y_test2), dim=0)
    return h_data_mixed,y_data_mixed,X_test_mixed,y_test_mixed

def cifar10_prepare_node_10_hard_linear_mix(p1=0.5, seed=42, augment=False):
    """ 
    p1 是高异质性数据抽样的比例，p2=1-p1 是均匀分布数据抽样的比例
    """
    # 获取原始数据
    h_data1, y_data1, X_test1, y_test1 = cifar10_prepare_node_10_hard(augment=augment)  # 大异质性，对应 p1
    h_data2, y_data2, X_test2, y_test2 = cifar10_prepare_node_10_hard_shuffled(augment=augment)  # 均匀，对应 p2
    
    p2 = 1 - p1
    
    # 设置随机种子
    np.random.seed(seed)

    h_data_mixed = []
    y_data_mixed = []

    # 依次对每个节点进行抽样
    for h1, y1, h2, y2 in zip(h_data1, y_data1, h_data2, y_data2):
        # 确定每个数据集的抽样数量
        len_h1 = len(h1)
        len_h2 = len(h2)
        
        # 计算抽样数量（不能整除时向下取整）
        sample_size_h1 = int(len_h1 * p1)
        sample_size_h2 = int(len_h2 * p2)
        
        # 从 h1, y1 和 h2, y2 中分别抽样
        indices_h1 = np.random.choice(len_h1, sample_size_h1, replace=False)
        indices_h2 = np.random.choice(len_h2, sample_size_h2, replace=False)
        
        h1_sampled = h1[indices_h1]
        y1_sampled = y1[indices_h1]
        h2_sampled = h2[indices_h2]
        y2_sampled = y2[indices_h2]

        # 将两个抽样结果拼接
        h_mixed = torch.cat([h1_sampled, h2_sampled], dim=0)
        y_mixed = torch.cat([y1_sampled, y2_sampled], dim=0)

        h_data_mixed.append(h_mixed)
        y_data_mixed.append(y_mixed)
    
    # 测试数据直接拼接
    X_test = torch.cat((X_test1, X_test2), dim=0)
    y_test = torch.cat((y_test1, y_test2), dim=0)
    
    return h_data_mixed, y_data_mixed, X_test, y_test