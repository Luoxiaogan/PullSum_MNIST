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

""" m = loadmat('MNIST_digits_2_4.mat')
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
    return total_loss """

from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

def new_train_PullSum( 
        n=5,
        A=None,
        B=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        batch_size=None,  # 新增参数
        show_graph=True
        ):

    lr = n * lr

    # 确保使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 确保训练数据在GPU上
    y_data = [y.to(device) for y in y_train_data]  # 确保训练标签在GPU上
    X_test_tensor = X_test_data.to(device)  # 确保测试数据在GPU上
    y_test_tensor = y_test_data.to(device)  # 确保测试标签在GPU上

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 确保模型在GPU上
    criterion = criterion_class().to(device)  # 确保损失函数在GPU上

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = PullSum(model_list=model_list, lr=lr, A=A, B=B, closure=closure)

    loss_history = []
    accuracy_history = []

    # 创建 tqdm 对象显示训练进度
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        epoch_loss = 0
        
        if batch_size is None:
            # 无批处理：直接使用整个数据进行训练
            loss = optimizer.step(closure)
            epoch_loss = loss
        else:
            # 使用 DataLoader 进行批处理
            for i, model in enumerate(model_list):
                model.train()
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(h_data[i], y_data[i]),
                    batch_size=batch_size,
                    shuffle=True
                )
                
                # 使用 tqdm 显示 batch 的进度
                batch_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Model {i+1}", leave=False)
                
                for batch_h_data, batch_y_data in batch_progress_bar:
                    def batch_closure():
                        model.zero_grad()
                        output = model(batch_h_data)
                        loss = criterion(output, batch_y_data)
                        loss.backward()
                        return loss
                    
                    # 执行优化步骤
                    loss = optimizer.step(batch_closure)
                    epoch_loss += loss.item()
                    batch_progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            epoch_loss = epoch_loss / len(train_loader)#标准化
                    
        loss_history.append(epoch_loss / len(model_list))
        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss_history[-1]:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history



def train_PullSum(
        n=5,
        A=None,
        B=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        show_graph=True
        ):
    
    lr = n * lr

    # 确保使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 确保训练数据在GPU上
    y_data = [y.to(device) for y in y_train_data]  # 确保训练标签在GPU上
    X_test_tensor = X_test_data.to(device)  # 确保测试数据在GPU上
    y_test_tensor = y_test_data.to(device)  # 确保测试标签在GPU上

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 确保模型在GPU上
    criterion = criterion_class().to(device)  # 确保损失函数在GPU上

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = PullSum(model_list=model_list, lr=lr, A=A, B=B, closure=closure)

    loss_history = []
    accuracy_history = []

    # 创建 tqdm 对象显示训练进度
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)
        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history


def train_PullDiag(
        n=5,
        A=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        show_graph=True):
    
    # 检查CUDA是否可用，并在可能的情况下使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 将训练数据移动到GPU
    y_data = [y.to(device) for y in y_train_data]  # 将训练标签移动到GPU
    X_test_tensor = X_test_data.to(device)  # 将测试数据移动到GPU
    y_test_tensor = y_test_data.to(device)  # 将测试标签移动到GPU

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 将模型移动到GPU
    criterion = criterion_class().to(device)  # 将损失函数移动到GPU

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = PullDiag(model_list=model_list, lr=lr, A=A, closure=closure)
    
    loss_history = []
    accuracy_history = []

    # 创建 tqdm 对象以显示训练进度
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)
        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'PullDiag, n={n}, lr={lr:.6f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history

def train_FRSD(
        n=5,
        A=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        beta=0.1,
        d=784,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        show_graph=True):
    
    # 检查CUDA是否可用，并在可能的情况下使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 将训练数据移动到GPU
    y_data = [y.to(device) for y in y_train_data]  # 将训练标签移动到GPU
    X_test_tensor = X_test_data.to(device)  # 将测试数据移动到GPU
    y_test_tensor = y_test_data.to(device)  # 将测试标签移动到GPU

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 将模型移动到GPU
    criterion = criterion_class().to(device)  # 将损失函数移动到GPU

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = FRSD(model_list=model_list, lr=lr, beta=beta, d=784 ,A=A, closure=closure)
    
    loss_history = []
    accuracy_history = []

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)

        # 检查是否有 inf 或 nan
        if np.isnan(loss) or np.isinf(loss):
            print(f"Stopping early due to inf/nan in loss at epoch {epoch + 1}")
            break  # 跳出循环

        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
    
    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'FRSD, n={n}, lr={lr:.6f}, beta={beta:.6f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history

def train_FROZEN(
        n=5,
        A=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        beta=0.1,
        d=784,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        show_graph=True):
    
    # 检查CUDA是否可用，并在可能的情况下使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 将训练数据移动到GPU
    y_data = [y.to(device) for y in y_train_data]  # 将训练标签移动到GPU
    X_test_tensor = X_test_data.to(device)  # 将测试数据移动到GPU
    y_test_tensor = y_test_data.to(device)  # 将测试标签移动到GPU

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 将模型移动到GPU
    criterion = criterion_class().to(device)  # 将损失函数移动到GPU

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = FROZEN(model_list=model_list, lr=lr, beta=beta, d=784 ,A=A, closure=closure)
    
    loss_history = []
    accuracy_history = []

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)

        # 检查是否有 inf 或 nan
        if np.isnan(loss) or np.isinf(loss):
            print(f"Stopping early due to inf/nan in loss at epoch {epoch + 1}")
            break  # 跳出循环

        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")
        
    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'FROZEN, n={n}, lr={lr:.6f}, beta={beta:.6f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history

def train_Diag_SDG(
        n=5,
        A=None,
        model_class=None,
        seed_for_model=42,
        criterion_class=nn.CrossEntropyLoss,
        epochs=10,
        lr=0.1,
        X_train_data=None,
        y_train_data=None,
        X_test_data=None,
        y_test_data=None,
        compute_accuracy=None,
        show_graph=True):
    
    # 检查CUDA是否可用，并在可能的情况下使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(A, dtype=torch.float32).to(device)
    h_data = [x.to(device) for x in X_train_data]  # 将训练数据移动到GPU
    y_data = [y.to(device) for y in y_train_data]  # 将训练标签移动到GPU
    X_test_tensor = X_test_data.to(device)  # 将测试数据移动到GPU
    y_test_tensor = y_test_data.to(device)  # 将测试标签移动到GPU

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]  # 将模型移动到GPU
    criterion = criterion_class().to(device)  # 将损失函数移动到GPU

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data[i])
            loss = criterion(output, y_data[i])
            loss.backward()
            total_loss += loss.item()
        return total_loss / len(model_list)
    
    optimizer = Diag_SGD(model_list=model_list, lr=lr, A=A, closure=closure)
    
    loss_history = []
    accuracy_history = []

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        loss = optimizer.step(closure)  # 调用优化器的 step 方法，并传入 closure
        loss_history.append(loss)

        # 检查是否有 inf 或 nan
        if np.isnan(loss) or np.isinf(loss):
            print(f"Stopping early due to inf/nan in loss at epoch {epoch + 1}")
            break  # 跳出循环

        accuracy = compute_accuracy(model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        accuracy_history.append(accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{loss:.10f}", accuracy=f"{100 * accuracy:.10f}%")

    if show_graph:
        # 绘制损失和准确率历史图
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

        plt.suptitle(f'Diag_SGD, n={n}, lr={lr:.6f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return loss_history, accuracy_history