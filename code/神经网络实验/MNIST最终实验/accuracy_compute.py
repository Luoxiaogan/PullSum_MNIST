import torch
from useful_model import *

def compute_accuracy_binary(model_list, X_test, y_test):
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        for i, model in enumerate(model_list):
            outputs = model(X_test)
            predicted = (outputs > 0.5).float().view(-1) # 二分类的阈值设置为0.5
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
    accuracy = correct / total
    return accuracy

def compute_accuracy_multiclass(model_list, X_test, y_test):
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        for i, model in enumerate(model_list):
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)  # 找到最大概率的类
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
    accuracy = correct / total
    return accuracy

import torch

def modelwise_compute_accuracy_with_average_model(model_class, model_list, X_test, y_test):
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device
    
    # Step 1: Compute the average of the parameters from all models
    avg_model = model_class().to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()}

    # 汇总所有模型的参数
    for model in model_list:
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        for key in sum_state_dict.keys():
            if key in state_dict:
                sum_state_dict[key] += state_dict[key].to(device)
            else:
                print(f"Warning: {key} not found in model's state dict.")

    # 计算平均值
    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    # 将平均参数加载到新模型中
    avg_model.load_state_dict(avg_state_dict)

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Step 2: Evaluate the new model's accuracy
    avg_model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        outputs = avg_model(X_test)  # 模型前向推理
        _, predicted = torch.max(outputs, 1)  # 获取最大概率的类
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)  # 计算准确率

    return accuracy



def compute_accuracy_with_average_model(model_list, X_test, y_test):
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = type(model_list[0])().to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()}

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

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Step 2: Evaluate the new model's accuracy
    avg_model.eval()
    with torch.no_grad():
        outputs = avg_model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    return accuracy

import torch
from torchvision.models import ResNet, resnet18

def compute_accuracy_with_average_model_resnet18(model_list, X_test, y_test):
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = resnet18(num_classes=10).to(device)  # 创建新的模型实例，并将其移动到同一设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()}

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

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Step 2: Evaluate the new model's accuracy
    avg_model.eval()
    with torch.no_grad():
        outputs = avg_model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    return accuracy

def compute_accuracy_with_average_model_mini_resnet18(model_list, X_test, y_test):
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = MiniResNet18_1().to(device)  # 创建新的缩小版模型实例，并将其移动到设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict
    sum_state_dict = {key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()}

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

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Step 2: Evaluate the new model's accuracy
    avg_model.eval()
    with torch.no_grad():
        outputs = avg_model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    return accuracy

def BJD_compute_accuracy_with_average_model_mini_resnet18(model_list, X_test, y_test):
    """ 半精度 """
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = MiniResNet18_1().to(device)  # 创建新的缩小版模型实例，并将其移动到设备上
    avg_state_dict = avg_model.state_dict()  # 获取新模型的状态字典

    # 初始化 sum_state_dict，使用与模型参数相同的数据类型
    sum_state_dict = {key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()}

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

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Step 2: Evaluate the new model's accuracy using mixed precision
    avg_model.eval()
    with torch.no_grad():
        # 使用 torch.cuda.amp 自动混合精度
        with torch.cuda.amp.autocast():
            outputs = avg_model(X_test)  # 推理时自动使用半精度
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    return accuracy

