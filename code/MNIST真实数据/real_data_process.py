import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat 

def load_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X / 255.0  # 归一化
    y = y.astype(int)  # 确保标签是整数

    # 选择数字3和8
    mask = (y == 3) | (y == 8)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # 将标签转换为二分类，3为1，8为-1
    y_filtered = np.where(y_filtered == 3, 1, -1)

    return X_filtered, y_filtered

def split_data(X, y, test_ratio=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
    return X_train, X_test, y_train, y_test

def distribute_data(X, y, n_nodes):
    """ 均分数据，如果数据长度不被n_nodes整除，丢弃剩余的部分 """
    n_samples_per_node = len(X) // n_nodes
    nodes_data = []
    for i in range(n_nodes):
        start = i * n_samples_per_node
        end = start + n_samples_per_node
        nodes_data.append((X[start:end], y[start:end]))
    return nodes_data

def compute_accuracy(x, X_test, y_test):
    # 计算均值 w
    w = np.mean(x, axis=0)

    # 计算 z = np.dot(X_test, w) 的向量化版本
    z = np.dot(X_test, w)

    # 计算概率
    prob = 1 / (1 + np.exp(-z))

    # 计算预测结果
    y_out = np.where(prob >= 0.5, 1, -1)

    # 计算正确和错误的预测次数
    right = np.sum(y_out == y_test)
    false = y_out.size - right

    # 计算准确率
    accuracy = right / (right + false)
    return accuracy