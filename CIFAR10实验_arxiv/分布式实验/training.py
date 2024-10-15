import torch
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

sys.path.append("/root/GanLuo/PullSum_MNIST/CIFAR10实验_arxiv")
from optimizer import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def get_first_batch(trainloader_list):
    h_data_train = []
    y_data_train = []

    # 遍历每个 trainloader
    for trainloader in trainloader_list:
        # 使用 tee 复制迭代器，不改变原始的迭代器
        loader_copy, trainloader = itertools.tee(trainloader, 2)

        # 从复制的迭代器中取第一个批次的数据
        first_batch = next(iter(loader_copy))

        # 分别保存 X 和 y
        h_data_train.append(first_batch[0])  # inputs (X)
        y_data_train.append(first_batch[1])  # labels (y)

    return h_data_train, y_data_train


def compute_accuracy(model_class, model_list, testloader, use_amp=False):
    # 使用 CrossEntropyLoss 作为默认损失函数
    criterion = nn.CrossEntropyLoss()

    # 确保模型在正确的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = model_class().to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {
        key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()
    }

    # 汇总所有模型的参数
    for model in model_list:
        state_dict = model.state_dict()
        for key in sum_state_dict.keys():
            sum_state_dict[key] += state_dict[key].to(device)

    # 计算平均值
    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    # 将平均参数加载到新模型中
    avg_model.load_state_dict(avg_state_dict)

    # Step 2: Evaluate the new model's loss and accuracy using test_loader
    avg_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):
                # 前向传播
                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            # 汇总损失
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 计算最终的平均损失和准确率
    average_loss = total_loss / (len(testloader))  # 两次标准化
    accuracy = correct / total

    return average_loss, accuracy


""" 
# GPT优化的版本, 可以选择混合精度计算

def train_PullSum(
    n=5,
    A=None,
    B=None,
    model_class=None,
    seed_for_model=42,
    epochs=10,
    lr=0.1,
    trainloader_list=None,
    testloader=None,
    show_graph=True,
    batch_size=128,
    csv_root=None,
    warm_up=False,
    use_amp=False,  # 添加参数
):
    torch.backends.cudnn.benchmark = True

    lr = n * lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]
    criterion = nn.CrossEntropyLoss().to(device)

    h_data_train, y_data_train = get_first_batch(trainloader_list)
    h_data_train = [tensor.to(device, non_blocking=True) for tensor in h_data_train]
    y_data_train = [tensor.to(device, non_blocking=True) for tensor in y_data_train]

    # 初始化 GradScaler
    scaler = GradScaler(enabled=use_amp)

    # 定义 closure_init 函数，用于优化器初始化
    def closure_init():
        total_loss = 0
        for i, model in enumerate(model_list):
            model.zero_grad()
            # 不使用混合精度
            output = model(h_data_train[i])
            loss = criterion(output, y_data_train[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)

    # 初始化优化器
    optimizer = PullSum(model_list=model_list, lr=lr, A=A, B=B, closure=closure_init)

    print("optimizer初始化成功!")

    # 定义 closure 函数，用于训练过程
    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            model.zero_grad()
            with autocast(enabled=use_amp):
                output = model(h_data_train[i])
                loss = criterion(output, y_data_train[i])
            total_loss += loss
        return total_loss / len(model_list)

    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        if warm_up:
            if epoch < n:
                lr = lr * (epoch + 1) / n
            else:
                lr = lr

        train_loss = 0

        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [data[0].to(device, non_blocking=True) for data in batch]
            labels = [data[1].to(device, non_blocking=True) for data in batch]

            h_data_train = inputs
            y_data_train = labels

            # 计算损失
            loss = closure()

            # 使用梯度缩放进行反向传播
            scaler.scale(loss).backward()

            # 对梯度进行反缩放
            scaler.unscale_(optimizer)

            # 进行优化器更新
            optimizer.step()

            # 更新 scaler
            scaler.update()

            train_loss += loss.item()

        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)

        average_loss, accuracy = compute_accuracy(
            model_class=model_class,
            model_list=model_list,
            testloader=testloader,
            use_amp=use_amp,
        )
        test_loss_history.append(average_loss)
        test_accuracy_history.append(accuracy)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.10f}",
            test_loss=f"{test_loss_history[-1]:.10f}",
            test_accuracy=f"{100 * test_accuracy_history[-1]:.10f}%",
        )

        # 将结果保存为 DataFrame，并存储为 CSV 文件
        if csv_root is not None:
            df = pd.DataFrame(
                {
                    "Epoch": list(range(1, epoch + 2)),
                    "Train_Loss": train_loss_history,
                    "Test_Loss": test_loss_history,
                    "Test_Accuracy": test_accuracy_history,
                }
            )
            csv_path = csv_root
            df.to_csv(csv_path, index=False)

    if show_graph:
        # 创建一个2行2列的子图布局
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, color="r", label="train_loss")
        plt.plot(test_loss_history, color="r", label="test_loss")
        plt.title("Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(test_accuracy_history, color="r")
        plt.title("Test Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.suptitle(f"PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()

    return train_loss_history, test_loss_history, test_accuracy_history """


def train_PullSum(
    n=5,
    A=None,
    B=None,
    model_class=None,
    seed_for_model=42,
    epochs=10,
    lr=0.1,
    trainloader_list=None,
    testloader=None,
    show_graph=True,
    batch_size=128,
    csv_root=None,
    warm_up=False,
):

    torch.backends.cudnn.benchmark = True

    lr = n * lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]
    criterion = nn.CrossEntropyLoss().to(device)

    h_data_train, y_data_train = get_first_batch(trainloader_list)
    h_data_train = [
        tensor.to(device, non_blocking=True) for tensor in h_data_train
    ]  # [tensor.to(device) for tensor in h_data_train]
    y_data_train = [
        tensor.to(device, non_blocking=True) for tensor in y_data_train
    ]  # [tensor.to(device) for tensor in y_data_train]

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data_train[i])
            loss = criterion(output, y_data_train[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)

    optimizer = PullSum(model_list=model_list, lr=lr, A=A, B=B, closure=closure)

    print("optimizer初始化成功!")

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:

        if warm_up:
            if epoch < n:
                lr = lr * (epoch + 1) / n
            else:
                lr = lr

        train_loss = 0

        # for batch_idx, batch in enumerate(zip(*[iter(loader) for loader in trainloader_list])): 直接遍历trainloader_list中的加载器，无需再使用iter和zip, 因为数据加载器中使用pin_memory=True
        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [
                data[0].to(device, non_blocking=True) for data in batch
            ]  # [data[0] for data in batch]
            labels = [
                data[1].to(device, non_blocking=True) for data in batch
            ]  # [data[1] for data in batch]
            h_data_train = inputs  # [tensor.to(device) for tensor in inputs]
            y_data_train = labels  # [tensor.to(device) for tensor in labels]
            loss = optimizer.step(closure=closure, lr=lr)
            train_loss += loss
        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)

        average_loss, accuracy = compute_accuracy(
            model_class=model_class, model_list=model_list, testloader=testloader
        )
        test_loss_history.append(average_loss)
        test_accuracy_history.append(accuracy)

        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.10f}",
            test_loss=f"{test_loss_history[-1]:.10f}",
            test_accuracy=f"{100 * test_accuracy_history[-1]:.10f}%",
        )

        # 将结果保存为 DataFrame，并存储为 CSV 文件
        if csv_root is not None:
            df = pd.DataFrame(
                {
                    "Epoch": list(range(1, epoch + 2)),
                    "Train_Loss": train_loss_history,
                    "Test_Loss": test_loss_history,
                    "Test_Accuracy": test_accuracy_history,
                }
            )
            csv_path = csv_root
            df.to_csv(csv_path, index=False)

    if show_graph:
        # 创建一个2行2列的子图布局
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, color="r", label="train_loss")
        plt.plot(test_loss_history, color="b", label="test_loss")
        plt.title("Loss Comparision")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(test_accuracy_history, color="b")
        plt.title("Test Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.suptitle(f"PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()

    return train_loss_history, test_loss_history, test_accuracy_history
