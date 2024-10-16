import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
import random


def distributed_cifar10_dataloaders(
    n,
    batch_size=128,
    root="/root/GanLuo/PullSum_MNIST/CIFAR10实验_arxiv/cifar-10-python/cifar-10-batches-py",
    seed=42,
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # 创建用于控制DataLoader随机性的生成器
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # 数据增强和预处理
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(
                32, padding=4
            ),  # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomRotation(15),  # Random rotation between -15 and 15 degrees
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Randomly change brightness, contrast, saturation, and hue
            transforms.RandomGrayscale(
                p=0.1
            ),  # Convert the image to grayscale with a probability of 0.1
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # Normalize using dataset statistics
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # 加载CIFAR10训练和测试集
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform_test
    )

    # 将训练集打乱并划分为n个子集
    trainloader_list = []
    total_train_size = len(trainset)  # CIFAR10 训练集大小是50000
    subset_size = total_train_size // n  # 每个节点获取的基本数据量
    remainder = total_train_size % n  # 计算余数数据

    # 获取乱序的索引列表
    indices = torch.randperm(total_train_size).tolist()

    start_idx = 0
    for i in range(n):
        end_idx = start_idx + subset_size
        if i < remainder:  # 前 remainder 个节点每个多分配一张图像
            end_idx += 1

        # 使用乱序索引创建每个节点的子集
        subset_indices = indices[start_idx:end_idx]
        subset = torch.utils.data.Subset(trainset, subset_indices)

        # 为每个节点创建DataLoader
        trainloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=15,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        trainloader_list.append(trainloader)

        # 更新索引区间
        start_idx = end_idx

    # 创建测试集 DataLoader (无需改变)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return trainloader_list, testloader

""" 根据这个函数, 写一个新的函数, 一般来说, 此时默认n=20

我有两组分布的数据, 一组数据上, 按照上面的函数, 正常的, 完全打乱了训练数据的均匀分布, 分布在20个节点上

还有一组数据是, CIFAR10有十类, 那么20个节点(0到19), 节点0和10保存第一类的数据, 节点1和11保存第二类的数据, 节点2和12保存第三类的数据, ..., 节点9和19保存第十类的数据

然后, 我根据函数的输入参数alpha, 作为第二组(大异质性数据)的比例, 即, 以概率p在第二组数据(每一个节点)随机抽样, 以概率1-p在第一组数据(每一个节点), 然后合并成新的数据, 然后把它做成trainloader """

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random


def hetero_distributed_cifar10_dataloaders(
    n=20,
    batch_size=128,
    root="/root/GanLuo/PullSum_MNIST/CIFAR10实验_arxiv/cifar-10-python/cifar-10-batches-py",
    seed=42,
    alpha=0.1,  # Proportion of heterogeneous data
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Function to initialize worker seeds
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Data augmentation and preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform_test
    )

    # Create the first dataset (fully shuffled and equally distributed among n nodes)
    total_train_size = len(trainset)  # CIFAR10 training set size is 50000
    indices = torch.randperm(total_train_size).tolist()
    subset_size = total_train_size // n
    remainder = total_train_size % n

    data_A_indices_list = []
    start_idx = 0
    for i in range(n):
        end_idx = start_idx + subset_size
        if i < remainder:
            end_idx += 1

        subset_indices = indices[start_idx:end_idx]
        data_A_indices_list.append(subset_indices)
        start_idx = end_idx

    # Create the second dataset (data divided by class labels among nodes)
    class_to_indices = {i: [] for i in range(10)}
    for idx, (_, target) in enumerate(trainset):
        class_to_indices[target].append(idx)

    # Shuffle indices within each class
    for c in range(10):
        random.shuffle(class_to_indices[c])

    data_B_indices_list = [[] for _ in range(n)]  # List of indices per node
    for c in range(10):
        class_indices = class_to_indices[c]
        num_samples = len(class_indices)
        node1 = c
        node2 = c + 10 if c + 10 < n else c + n // 2  # Adjust if n < 20

        num_samples_per_node = num_samples // 2
        remainder = num_samples % 2

        start_idx = 0
        end_idx = start_idx + num_samples_per_node + (1 if remainder > 0 else 0)
        data_B_indices_list[node1].extend(class_indices[start_idx:end_idx])

        start_idx = end_idx
        end_idx = num_samples
        data_B_indices_list[node2 % n].extend(class_indices[start_idx:end_idx])

    # Create DataLoaders for each node
    trainloader_list = []
    for i in range(n):
        data_A_indices = data_A_indices_list[i]
        data_B_indices = data_B_indices_list[i]

        len_A = len(data_A_indices)
        len_B = len(data_B_indices)

        total_size = len_A  # Total data per node is based on the first dataset

        n_B_i = int(alpha * total_size)
        n_A_i = total_size - n_B_i

        # Ensure we do not sample more than available
        n_B_i = min(n_B_i, len_B)
        n_A_i = min(n_A_i, len_A)

        # Randomly sample data from both datasets
        sampled_A_indices = random.sample(data_A_indices, n_A_i)
        sampled_B_indices = random.sample(data_B_indices, n_B_i)

        # Combine sampled data
        combined_indices = sampled_A_indices + sampled_B_indices

        # Create subset and DataLoader for the node
        subset = torch.utils.data.Subset(trainset, combined_indices)
        trainloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=15,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        trainloader_list.append(trainloader)

    # Create test DataLoader (unchanged)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return trainloader_list, testloader
