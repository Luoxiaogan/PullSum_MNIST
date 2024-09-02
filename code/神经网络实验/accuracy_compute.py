import torch

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