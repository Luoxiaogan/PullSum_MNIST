import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from useful_functions import *
from optimizer import *
from model import *
from MNIST_data_process import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from training import *
from mlxtend.data import mnist_data
from accuracy_compute import *
from data_preparation_easy import *
from data_preparation_hard import *
from network_func import *
import optuna

def search_PullSum(
        n=5,
        A=None,
        B=None,
        h_data=None,
        y_data=None,
        X_test=None,
        y_test=None,
        epochs=100,
        model_class=MNISTClassifier_2layer_2,
        lr_min=1e-4,
        lr_max=1,
        n_trials=10
        ):
    def objective(trial):
        # 缩小 lr 的搜索范围，在已找到的最佳值附近搜索
        lr = trial.suggest_loguniform('lr',lr_min ,lr_max)

        try:
            Sum_l0, Sum_a0 = train_PullSum(
                n=n,
                A=A,
                B=B,
                model_class=model_class,
                seed_for_model=49,
                criterion_class=nn.CrossEntropyLoss,
                epochs=epochs,
                lr=lr,
                X_train_data=h_data,
                y_train_data=y_data,
                X_test_data=X_test,
                y_test_data=y_test,
                compute_accuracy=compute_accuracy_with_average_model,
                show_graph=False
            )

            # 检查是否包含 inf 或 nan
            if np.isnan(Sum_l0).any() or np.isinf(Sum_l0).any():
                print(f"Trial failed due to inf/nan in loss. lr: {lr}")
                return -np.inf  # 返回一个非常低的值

            # 返回最终的准确率
            return Sum_a0[-1]

        except Exception as e:
            print(f"Trial failed with exception: {e}")
            return -np.inf  # 若发生异常，返回一个非常低的值

    # 创建一个优化器并使用缩小后的搜索空间进行优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")


def search_PullDiag(
        n=5,
        A=None,
        h_data=None,
        y_data=None,
        X_test=None,
        y_test=None,
        epochs=100,
        model_class=MNISTClassifier_2layer_2,
        lr_min=1e-4,
        lr_max=1,
        n_trials=10
        ):
    def objective(trial):
        # 缩小 lr 的搜索范围，在已找到的最佳值附近搜索
        lr = trial.suggest_loguniform('lr',lr_min ,lr_max)

        try:
            Sum_l0, Sum_a0 = train_PullDiag(
                n=n,
                A=A,
                model_class=model_class,
                seed_for_model=49,
                criterion_class=nn.CrossEntropyLoss,
                epochs=epochs,
                lr=lr,
                X_train_data=h_data,
                y_train_data=y_data,
                X_test_data=X_test,
                y_test_data=y_test,
                compute_accuracy=compute_accuracy_with_average_model,
                show_graph=False
            )

            # 检查是否包含 inf 或 nan
            if np.isnan(Sum_l0).any() or np.isinf(Sum_l0).any():
                print(f"Trial failed due to inf/nan in loss. lr: {lr}")
                return -np.inf  # 返回一个非常低的值

            # 返回最终的准确率
            return Sum_a0[-1]

        except Exception as e:
            print(f"Trial failed with exception: {e}")
            return -np.inf  # 若发生异常，返回一个非常低的值

    # 创建一个优化器并使用缩小后的搜索空间进行优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")
def search_PullFRSD(
    n=5,
    A=None,
    h_data=None,
    y_data=None,
    X_test=None,
    y_test=None,
    epochs=100,
    model_class=MNISTClassifier_2layer_2,
    lr_min=1e-4,
    lr_max=1,
    beta_min=1e-4,
    beta_max=1,
    loss_threshold=1.0,  # 设定一个合理的 loss 阈值
    n_trials=10
):
    def objective(trial):
        # 缩小 lr 的搜索范围，在已找到的最佳值附近搜索
        lr = trial.suggest_loguniform('lr', lr_min, lr_max)
        beta = trial.suggest_loguniform('beta', beta_min, beta_max)

        try:
            # 调用训练函数
            Sum_l0, Sum_a0 = train_FRSD(
                n=n,
                A=A,
                model_class=model_class,
                seed_for_model=49,
                criterion_class=nn.CrossEntropyLoss,
                epochs=epochs,
                lr=lr,
                beta=beta,
                X_train_data=h_data,
                y_train_data=y_data,
                X_test_data=X_test,
                y_test_data=y_test,
                compute_accuracy=compute_accuracy_with_average_model,
                show_graph=False
            )

            # 检查是否包含 inf 或 nan
            if np.isnan(Sum_l0).any() or np.isinf(Sum_l0).any():
                print(f"Trial failed due to inf/nan in loss. lr: {lr}")
                return -np.inf  # 返回一个非常低的值

            # 计算最后十个准确率的平均值
            avg_accuracy = np.mean(Sum_a0[-10:])
            avg_loss = np.mean(Sum_l0[-10:])

            # 根据 loss 来调整返回值，如果 loss 太大，准确率再高也不能接受
            if avg_loss > loss_threshold:
                print(f"Trial failed due to high loss. lr: {lr}, avg_loss: {avg_loss}")
                return -np.inf  # 罚分，返回一个非常低的值

            # 返回最后十个准确率的平均值作为优化目标
            return avg_accuracy

        except Exception as e:
            print(f"Trial failed with exception: {e}")
            return -np.inf  # 若发生异常，返回一个非常低的值

    # 创建一个优化器并使用缩小后的搜索空间进行优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")
