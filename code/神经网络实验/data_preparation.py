import torch
from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split

#获取node=5的分类数据
def prepare_node_5():
    X, y = mnist_data()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    h_data = [[] for _ in range(5)] 
    y_data = [[] for _ in range(5)]
    for i in range(len(y_train)):
        label = y_train[i].item()
        group_index = label // 2  # 根据标签分组，0-1 -> 0, 2-3 -> 1, 4-5 -> 2, 6-7 -> 3, 8-9 -> 4
        h_data[group_index].append(X_train[i])
        y_data[group_index].append(y_train[i])
    h_data = [torch.stack(group_data) for group_data in h_data]
    y_data = [torch.tensor(group_labels) for group_labels in y_data]
    return h_data,y_data,X_test,y_test

#获取node=10的分类数据
def prepare_node_10():
    X, y = mnist_data()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    X_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 X_train
    y_train_classified = [[] for _ in range(10)]  # 用于存储不同类别的 y_train
    for i in range(len(y_train)):
        label = y_train[i].item()
        X_train_classified[label].append(X_train[i])
        y_train_classified[label].append(y_train[i])
    X_train_list = [torch.stack(class_data) for class_data in X_train_classified]
    y_train_list = [torch.tensor(class_labels) for class_labels in y_train_classified]
    return X_train_list,y_train_list,X_test,y_test

