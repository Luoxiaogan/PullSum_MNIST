import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from useful_functions import *
from optimizer import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from concurrent.futures import ProcessPoolExecutor
from IPython.display import clear_output
import time
from resnet_model import old_ResNet18, new_ResNet18, init_params


def CA(model_class,model_list, X_test, y_test, batch_size=100):
    # 确保 X_test 和 y_test 在相同的设备上
    device = next(model_list[0].parameters()).device

    # Step 1: Compute the average of the parameters from all models
    avg_model = model_class().to(device)  # 创建新的模型实例，并将其移动到同一设备上
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

    # Step 2: Create test DataLoader with batch_size=100
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Evaluate the new model's accuracy using test_loader
    avg_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = avg_model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy

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

def six_GPU_train_PullSum( 
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
        batch_size=None,  # 新增参数
        show_graph=True
        ):

    lr = n * lr

    # cuda0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将 A, B, X_test_tensor 和 y_test_tensor 放在 cuda0
    A = torch.tensor(A, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)
    X_test_tensor = X_test_data.to(device)
    y_test_tensor = y_test_data.to(device)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size = 100,
        shuffle=False,
        num_workers=0
    )

    # 将 h_data 和 y_data 放在 cuda1 到 cuda5
    h_data = [data.to(f"cuda:{i + 1}") for i, data in enumerate(X_train_data)]
    y_data = [label.to(f"cuda:{i + 1}") for i, label in enumerate(y_train_data)]

    # 在各自的 GPU 上构建训练的 DataLoader
    train_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(h_data[i], y_data[i]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        for i in range(len(h_data))
    ]

    # 将模型放到 cuda1 到 cuda5
    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(f"cuda:{i + 1}") for i in range(5)]

    criterion_list = [criterion_class().to(f"cuda:{i}") for i in range(6)]

    def closure():
        total_loss = 0.0
        for i, model in enumerate(model_list):
            model.zero_grad()
            for batch_h, batch_y in train_loaders[i]:
                batch_h = batch_h.to(f"cuda:{i + 1}")
                batch_y = batch_y.to(f"cuda:{i + 1}")
                output = model(batch_h)
                loss = criterion_list[i + 1](output, batch_y)
                loss.backward()
                total_loss += loss.to('cuda:0').item()
            total_loss = total_loss / len(train_loaders[i])
        return total_loss / len(model_list)

    h_data_train = [data.clone().to(device) for data in h_data]
    y_data_train = [label.clone().to(device) for label in y_data]

    # 拼接数据
    X_data_for_accuracy_compute = torch.cat(h_data_train, dim=0)
    y_data_for_accuracy_compute = torch.cat(y_data_train, dim=0)
    train_accuracy_dataset = TensorDataset(X_data_for_accuracy_compute, y_data_for_accuracy_compute)
    train_accuracy_compute_loader = DataLoader(
        train_accuracy_dataset,
        batch_size=100,
        shuffle=False, 
        num_workers=0
    )
    
    optimizer = PullSum_for_try(model_list=model_list, lr=lr, A=A, B=B, closure=closure)
    #后面的batch中不需要更改optimizer, 只需要更改h_data_train和y_data_train的定义就行了

    print("optimizer初始化成功!")

    training_loss_history = []
    training_accuracy_history = []
    test_accuracy_history = []

    def closure_for_batch(batch_h_list, batch_y_list):
        loss_list = []
        for i, model in enumerate(model_list):
            model.zero_grad()
            output = model(batch_h_list[i])
            loss = criterion_list[i + 1](output, batch_y_list[i])
            loss.backward()
            loss_list.append(loss.unsqueeze(0).to('cuda:0'))
        total_loss = torch.cat(loss_list).sum().item()
        return total_loss / len(model_list)
    
    def compute_avg_model():
        new_model = model_class().to("cuda:0")
        with torch.no_grad():
            for new_param, *all_params in zip(new_model.parameters(), *(m.parameters() for m in model_list)):
                avg_param = torch.zeros_like(new_param).to("cuda:0")
                for param in all_params:
                    param_cuda0 = param.to("cuda:0") 
                    avg_param += param_cuda0
                    del param_cuda0
                avg_param /= len(model_list)
                new_param.data.copy_(avg_param)
                del avg_param
        return new_model
    
    def c_accuracy(model=None,loader=None,device='cuda:0'):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_h, batch_y in loader:
                batch_h = batch_h.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_h)
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy 

    def average_accuracy(model_list, loader, device_prefix='cuda', start_idx=1, num_gpus=5):
        total_accuracy = 0.0

        # 遍历每个模型和设备
        for i in range(start_idx, num_gpus + start_idx):
            device = f'{device_prefix}:{i}'
            accuracy = c_accuracy(model=model_list[i - start_idx], loader=loader, device=device)
            total_accuracy += accuracy

        # 计算平均准确率
        avg_accuracy = total_accuracy / num_gpus
        return avg_accuracy      

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        epoch_loss = 0

        if batch_size is None:
            loss = optimizer.step(closure)
            epoch_loss = loss
        else:
            for batch_tuple in zip(*train_loaders):
                batch_h_list = []
                batch_y_list = []
                for i in range(5):
                    batch_h, batch_y = batch_tuple[i]
                    batch_h_list.append(batch_h)
                    batch_y_list.append(batch_y)
                loss = optimizer.step(lambda: closure_for_batch(batch_h_list, batch_y_list))
                epoch_loss += loss
            epoch_loss = epoch_loss / len(train_loaders[0])
        training_loss_history.append(epoch_loss)

        #avg_model = compute_avg_model()
        #test_accuracy = c_accuracy(model=avg_model,loader=test_loader)
        #training_accuracy = c_accuracy(model=avg_model,loader=train_accuracy_compute_loader)
        test_accuracy = average_accuracy(model_list=model_list, loader=test_loader, start_idx=1, num_gpus=5)#c_accuracy(model=model_list[4],loader=test_loader,device='cuda:5')
        training_accuracy = average_accuracy(model_list=model_list, loader=train_accuracy_compute_loader, start_idx=1, num_gpus=5)#c_accuracy(model=model_list[4],loader=train_accuracy_compute_loader,device='cuda:5')
        test_accuracy_history.append(test_accuracy)
        training_accuracy_history.append(training_accuracy)

        #avg_model.to('cpu')
        #del avg_model

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, training_loss=f"{training_loss_history[-1]:.10f}", training_accuracy=f"{100*training_accuracy_history[-1]:.10f}%", test_accuracy=f"{100*test_accuracy_history[-1]:.10f}%")
    
    if show_graph:
        plt.subplot(1,2,1)
        plt.plot(training_loss_history, color='r')
        plt.title('Train Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(training_accuracy_history, color='r',label='training')
        plt.plot(test_accuracy_history, color='b',label='test')
        plt.title('Accuracy Comparision')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.show()
    return training_loss_history,training_accuracy_history, test_accuracy_history

def new_new_train_PullSum( 
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

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size = 100,
        shuffle=False,
        num_workers=0
    )

    train_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(h_data[i], y_data[i]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        for i in range(len(h_data))
    ]

    torch.manual_seed(seed_for_model)
    model_list = [model_class().to(device) for _ in range(n)]
    criterion = criterion_class().to(device)

    def closure():
        total_loss = 0.0
        for i, model in enumerate(model_list):
            model.zero_grad()
            for batch_h, batch_y in train_loaders[i]:
                output = model(batch_h)
                loss = criterion(output, batch_y)
                loss.backward()
                total_loss += loss.item()
            total_loss = total_loss / len(train_loaders[i])
        return total_loss / len(model_list)

    X_data_for_accuracy_compute = torch.cat(h_data, dim=0)  # 在第0维上连接
    y_data_for_accuracy_compute = torch.cat(y_data, dim=0)  # 在第0维上连接
    train_accuracy_dataset = TensorDataset(X_data_for_accuracy_compute, y_data_for_accuracy_compute)

    train_accuracy_compute_loader = DataLoader(
        train_accuracy_dataset,
        batch_size=100,
        shuffle=False, 
        num_workers=0
    )

    optimizer = PullSum_for_try(model_list=model_list, lr=lr, A=A, B=B, closure=closure)

    print("optimizer初始化成功!")

    training_loss_history = []
    training_accuracy_history = []
    test_accuracy_history = []

    def closure_for_batch(batch_h_list, batch_y_list):
        loss_list = []
        for i, model in enumerate(model_list):
            model.zero_grad()
            output = model(batch_h_list[i])
            loss = criterion(output, batch_y_list[i])
            loss.backward()
            loss_list.append(loss.unsqueeze(0))
        total_loss = torch.cat(loss_list).sum().item()
        return total_loss / len(model_list)
    
    def c_accuracy(model=None,loader=None):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_h, batch_y in loader:
                output = model(batch_h)
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy 

    # 创建 tqdm 对象显示训练进度
    #progress_bar = tqdm(range(epochs), desc="Training Progress")
    print("开始训练!")

    for epoch in range(epochs):
        epoch_loss = 0
        
        if batch_size is None:
            # 无批处理：直接使用整个数据进行训练
            loss = optimizer.step(closure)
            epoch_loss = loss
        else:# 批处理
            j=0
            for batch_tuple in zip(*train_loaders):
                batch_h_list = []
                batch_y_list = []
                for i in range(5):
                    batch_h, batch_y = batch_tuple[i]
                    batch_h_list.append(batch_h)
                    batch_y_list.append(batch_y)
                loss = optimizer.step(lambda: closure_for_batch(batch_h_list, batch_y_list))
                epoch_loss += loss
                print(f"batch:{j}成功！")
                j=j+1
            epoch_loss = epoch_loss / len(train_loaders[0])
        
        print("训练完了所有的batch")
        clear_output(wait=True)

        training_loss_history.append(epoch_loss)
        test_accuracy = c_accuracy(model=model_list[4],loader=test_loader)
        training_accuracy = c_accuracy(model=model_list[4],loader=train_accuracy_compute_loader)
        test_accuracy_history.append(test_accuracy)
        training_accuracy_history.append(training_accuracy)

        print("完成正确率计算!")
        clear_output(wait=True)
        print("hao")
        print(epoch_loss,test_accuracy,training_accuracy)


        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        #progress_bar.set_postfix(epoch=epoch + 1, loss=f"{training_loss_history[-1]:.10f}",trian_accuracy=f"{100 * training_accuracy:.10f}%", test_accuracy=f"{100 * test_accuracy:.10f}%")
    
    print("训练完了所有的epoch")
    
    if show_graph:
        # 创建一个2行2列的子图布局
        plt.subplot(1, 2, 1)
        plt.plot(training_loss_history, color='r')
        plt.title('Train Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(training_accuracy_history, color='b')
        plt.title('Train Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()

        plt.subplot(1, 2, 1)
        plt.plot(test_accuracy_history, color='b')
        plt.title('Test Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(training_accuracy_history, color='blue', label='Train')
        plt.plot(test_accuracy_history, color='red', label='Test')
        plt.title('Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()


    return training_loss_history, test_accuracy_history


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
        #compute_accuracy=None,
        batch_size=None,  # 新增参数
        show_graph=True,
        try_init=False
        ):
    
    compute_accuracy = lambda model_class, model_list, X_test, y_test, batch_size=100: CA(model_class=model_class, model_list=model_list, X_test=X_test, y_test=y_test, batch_size=batch_size)

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
    model_list = [model_class().to(device) for _ in range(n)]
    criterion = criterion_class().to(device)  # 确保损失函数在GPU上

    X_data_for_accuracy_compute = torch.cat(h_data, dim=0)  # 在第0维上连接
    y_data_for_accuracy_compute = torch.cat(y_data, dim=0)  # 在第0维上连接

    #准备如果使用batch的话，的loader
    h_data_batches = []
    y_data_batches = []
    train_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(h_data[i], y_data[i]),batch_size=batch_size,shuffle=True) for i in range(n)]
    train_loader_iters = [iter(loader) for loader in train_loaders]
    for _ in range(len(train_loaders[0])):  # 所有模型的数据集长度相同
        h_data_batch = []
        y_data_batch = []
        for i in range(n):
            batch_h_data, batch_y_data = next(train_loader_iters[i])  # 获取该模型当前 batch
            h_data_batch.append(batch_h_data)
            y_data_batch.append(batch_y_data)

        h_data_batches.append(h_data_batch)
        y_data_batches.append(y_data_batch)

    h_data_train = h_data_batches[0]
    y_data_train = y_data_batches[0]

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

    if try_init:
        for model in model_list:
            init_params(model)
        print("自定义的模型初始化成功!")
    clear_output(wait=True)

    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    # 创建 tqdm 对象显示训练进度
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        epoch_loss = 0
        
        if batch_size is None:
            # 无批处理：直接使用整个数据进行训练
            loss = optimizer.step(closure)
            epoch_loss = loss
        else:# 批处理
            for batch_h_data, batch_y_data in zip(h_data_batches, y_data_batches):
                h_data_train = batch_h_data
                y_data_train = batch_y_data
                loss = optimizer.step(closure)
                epoch_loss += loss
            epoch_loss = epoch_loss/len(train_loaders[0])#标准化
        train_loss_history.append(epoch_loss / len(model_list))#另一个标准化
        test_accuracy = compute_accuracy(model_class,model_list, X_test_tensor, y_test_tensor)  # 计算测试集上的准确率
        test_accuracy_history.append(test_accuracy)
        train_accuracy = compute_accuracy(model_class,model_list, X_data_for_accuracy_compute, y_data_for_accuracy_compute)  # 计算训练集上的准确率
        train_accuracy_history.append(train_accuracy)

        # 使用 set_postfix 方法来更新显示当前的 loss 和 accuracy
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{train_loss_history[-1]:.10f}",trian_accuracy=f"{100 * train_accuracy:.10f}%", test_accuracy=f"{100 * test_accuracy:.10f}%")

        yield train_loss_history, train_accuracy_history, test_accuracy_history
    
    if show_graph:
        # 创建一个2行2列的子图布局
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, color='r')
        plt.title('Train Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy_history, color='b')
        plt.title('Train Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()

        plt.subplot(1, 2, 1)
        plt.plot(test_accuracy_history, color='b')
        plt.title('Test Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy_history, color='blue', label='Train')
        plt.plot(test_accuracy_history, color='red', label='Test')
        plt.title('Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle(f'PullSum, n={n}, lr={lr:.6f}, batch_size={batch_size}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # 调整顶部边距
        plt.show()


    return train_loss_history, train_accuracy_history, test_accuracy_history



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