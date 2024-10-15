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
