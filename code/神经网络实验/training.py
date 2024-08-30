import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from useful_functions import *
from optimizer import *
from model import *
from MNIST_data_process import *
import matplotlib.pyplot as plt
from tqdm import tqdm

m = loadmat('MNIST_digits_2_4.mat')
X_train = m['X_train']
y_train = m['y_train'].transpose().reshape(-1)
X_test = m["X_test"]
y_test = m["y_test"].transpose().reshape(-1)
y_train = np.where(y_train == -1, 0, 1)
y_test = np.where(y_test == -1, 0, 1)

def custom_loss(output, target, model, rho=0.001):
    # 二分类交叉熵损失
    bce_loss = nn.BCELoss()(output, target)

    # 计算L2正则化项
    l2_reg = torch.tensor(0.0)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    
    # 总损失 = BCE损失 + L2正则化项
    total_loss = bce_loss + rho * l2_reg
    return total_loss

def train_PullSum(n=5,d=784,A=None,B=None,seed_for_model=42,criterion=nn.BCELoss(),epochs=10,lr=0.1,rho=0.1):
    lr=n*lr
    nodes_data = distribute_data(X=X_train, y=y_train, n_nodes=n)
    h = np.stack([i for i, j in nodes_data], axis=0)  # 这将创建一个 (n, L, d) 形状的数组
    y = np.stack([j for i, j in nodes_data], axis=0)  # 这将创建一个 (n, L) 形状的数组

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)
    h_data = torch.tensor(h, dtype=torch.float32).to(device)
    y_data = torch.tensor(y, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    torch.manual_seed(seed_for_model)
    model_list=[SimpleNN() for _ in range(n)]
    criterion=criterion

    def closure():
        total_loss=0
        for i,model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output=model(h_data[i])
            loss=criterion(output.view(-1),y_data[i])
            loss.backward()
            total_loss+=loss.item()
        return total_loss/(len(model_list))
    
    def compute_accuracy(model_list, X_test, y_test):
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
    
    optimizer = PullSum(model_list=model_list, lr=lr, A=A, B=B,closure=closure)
    # 假设 epochs 和其他变量已经定义
    loss_history = []
    accuracy_history = []

    loss_history = []
    accuracy_history = []

    # 创建 tqdm 对象
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)
        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='r')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, color='b')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.suptitle(f'PullSum,n={n},lr={lr}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return loss_history,accuracy_history

def train_PullDiag(n=5,d=784,A=None,seed_for_model=42,criterion=nn.BCELoss(),epochs=10,lr=0.1,rho=0.1):
    nodes_data = distribute_data(X=X_train, y=y_train, n_nodes=n)
    h = np.stack([i for i, j in nodes_data], axis=0)  # 这将创建一个 (n, L, d) 形状的数组
    y = np.stack([j for i, j in nodes_data], axis=0)  # 这将创建一个 (n, L) 形状的数组

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    h_data = torch.tensor(h, dtype=torch.float32).to(device)
    y_data = torch.tensor(y, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    torch.manual_seed(seed_for_model)
    model_list=[SimpleNN() for _ in range(n)]
    criterion=criterion

    def closure():
        total_loss=0
        for i,model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output=model(h_data[i])
            loss=criterion(output.view(-1),y_data[i])
            loss.backward()
            total_loss+=loss.item()
        return total_loss/(len(model_list))
    
    def compute_accuracy(model_list, X_test, y_test):
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
    
    optimizer = PullDiag(model_list=model_list, lr=lr, A=A,closure=closure)
    # 假设 epochs 和其他变量已经定义
    loss_history = []
    accuracy_history = []

    loss_history = []
    accuracy_history = []

    # 创建 tqdm 对象
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)
        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='r')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, color='b')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.suptitle(f'PullDiag,n={n},lr={lr}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return loss_history,accuracy_history