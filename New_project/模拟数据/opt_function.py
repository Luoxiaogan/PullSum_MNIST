import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from useful_functions import *
import copy


def pull_sum(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    grad_f_bar_func,
    d=784,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
    mg=1,
    warm_up = 10,
):
    """pull sum"""
    n = A.shape[0]
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n / mg * np.random.normal(
        size=(n, d)
    )
    V = np.eye(n)
    for _ in range(warm_up):
        V = A @ V
    v = copy.deepcopy(g)

    pi = get_left_perron(A).reshape((1, n))
    A0 = np.ones((n, 1)) @ pi @ A
    gradient_f_bar_x_val = grad_f_bar_func((A0 @ x).reshape(x.shape), y, h, rho=rho)
    gradient_history = [np.linalg.norm(gradient_f_bar_x_val)]
    v_history = [np.linalg.norm(v)]
    x_history = [x.mean()]
    diff = [np.linalg.norm(A0-V)]

    for _ in range(max_it):
        x = A @ x - lr * v
        pre_g = copy.deepcopy(g)
        g = grad_func(x, y, h, rho=rho).reshape(
            x.shape
        ) + sigma_n / mg * np.random.normal(size=(n, d))
        V = A @ V
        D = np.diag(1 / np.diag(V))
        v = A @ v + D @ (g - pre_g)

        gradient_history.append(np.linalg.norm(g))
        v_history.append(np.linalg.norm(v))
        x_history.append(x.mean())
        diff.append(np.linalg.norm(A0-V))
    
    result_df = pd.DataFrame({
        'gradient_history': gradient_history,
        'v_history': v_history,
        'x_history': x_history,
        'diff': diff
    })
    return result_df


def pull_diag(
    A,
    init_x,
    h_data,
    y_data,
    grad_func,
    grad_f_bar_func,
    d=784,
    rho=0.1,
    lr=0.1,
    sigma_n=0.1,
    max_it=200,
    mg=1,
):
    """pull diag"""
    n = A.shape[0]
    h, y = copy.deepcopy(h_data), copy.deepcopy(y_data)
    x = copy.deepcopy(init_x)
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape) + sigma_n / mg * np.random.normal(
        size=(n, d)
    )
    w = np.linalg.inv(np.diag(np.diag(W))) @ g
    v = copy.deepcopy(g)

    pi = get_left_perron(A).reshape((1, n))
    A0 = np.ones((n, 1)) @ pi @ A
    gradient_f_bar_x_val = grad_f_bar_func((A0 @ x).reshape(x.shape), y, h, rho=rho)
    gradient_history = [np.linalg.norm(gradient_f_bar_x_val)]
    Ls = [1 / min(np.diag(A))]
    v_history = [np.linalg.norm(v)]
    x_history = [x.mean()]

    for i in range(max_it):
        W = A @ W
        x = A @ x - lr * v
        gradient = grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape) + sigma_n / mg * np.random.normal(size=(n, d))
        v = A @ v + np.linalg.inv(np.diag(np.diag(W))) @ g - w
        w = (
            np.linalg.inv(np.diag(np.diag(W))) @ g
        )  # 这一步计算的w是下一步用到的w，因此程序没有问题
        g1 = grad_f_bar_func((A0 @ x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1))
        Ls.append(1 / min(np.diag(W)))
        v_history.append(np.linalg.norm(v))
        x_history.append(x.mean())

    result_df = pd.DataFrame({
        'gradient_history': gradient_history,
        'v_history': v_history,
        'x_history': x_history
    })
    return result_df
