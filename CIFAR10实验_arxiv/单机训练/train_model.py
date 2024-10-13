import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import time

def train_model(net, trainloader, testloader, device, EPOCH=100, LR=1e-2, optimizer_type='SGD', csv_path='metrics.csv'):
    """
    训练模型的函数。

    参数：
    - net: 要训练的神经网络模型。
    - trainloader: 训练数据的DataLoader。
    - testloader: 测试数据的DataLoader。
    - device: 计算设备（'cpu'或'cuda'）。
    - EPOCH: 训练的轮数。
    - LR: 学习率。
    - optimizer_type: 优化器类型，'SGD'或'momentum'。
    - csv_path: 保存指标的CSV文件路径。
    """
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 根据优化器类型选择优化器
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=LR)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    else:
        raise ValueError("optimizer_type must be 'SGD' or 'Adam'")

    # 学习率调度器
    milestones = [int(EPOCH * 0.5), int(EPOCH * 0.75)]  # 根据EPOCH调整里程碑
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 用于保存指标的列表
    Train_Accuracy = []
    Train_Loss = []
    Test_Accuracy = []
    Test_Loss = []

    # 开始训练
    for epoch in range(EPOCH):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # 训练阶段
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

        scheduler.step()

        train_loss = train_loss / len(trainloader)
        train_accu = 100. * correct / total

        # 测试阶段
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().item()

        test_loss = test_loss / len(testloader)
        test_accu = 100. * correct / total

        # 保存指标
        Train_Accuracy.append(train_accu)
        Train_Loss.append(train_loss)
        Test_Accuracy.append(test_accu)
        Test_Loss.append(test_loss)

        # 将指标保存到CSV文件
        metrics_df = pd.DataFrame({
            'Epoch': list(range(1, epoch + 2)),
            'Train_Loss': Train_Loss,
            'Train_Accuracy': Train_Accuracy,
            'Test_Loss': Test_Loss,
            'Test_Accuracy': Test_Accuracy
        })
        metrics_df.to_csv(csv_path, index=False)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60.0

        print(f"[Epoch:{epoch + 1}/{EPOCH}] "
              f"Train Loss: {train_loss:.3f} | Train Acc: {train_accu:.3f}% "
              f"Test Loss: {test_loss:.3f} | Test Acc: {test_accu:.3f}% "
              f"Time: {elapsed_time:.2f} min")

    print(f"Training Finished, Total EPOCH={EPOCH}")
