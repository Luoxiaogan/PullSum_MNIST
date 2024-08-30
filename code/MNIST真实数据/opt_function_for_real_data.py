import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from useful_functions import *

def new_pull_sum_real_data(A,B_index, init_x, h_data, y_data, X_test, y_test, grad_func, loss_func, grad_f_bar_func, accuracy_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    lr=n*lr#pull sum需要补偿n倍的学习率 
    h, y = h_data, y_data
    x = init_x
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g 
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    B=get_B(A,B_index,n)
    correction_vec=np.ones(n)
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    gradient_history=[np.linalg.norm(gradient_f_bar_x_val)]
    v_history=[np.linalg.norm(v)]
    loss_history=[loss_func(np.dot(pi,x), y=y, h=h, rho=rho)]
    accuracy_history=[accuracy_func(A0@x, X_test, y_test)]
    x_mean=np.mean(x,axis=0)
    cs_error=[np.linalg.norm(x-x_mean)]

    for i in range(max_it):
        
        correction_vec=A.T@correction_vec 
        x = A @ x - lr * np.diag(1/correction_vec) @ v 
        pre_g = g
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g 
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1))
        v_history.append(np.linalg.norm(np.diag(1/correction_vec) @v))
        loss_history.append(loss_func(np.dot(pi,x), y=y, h=h, rho=rho))
        accuracy_history.append(accuracy_func(A0@x, X_test, y_test))
        x_mean=np.mean(x,axis=0)
        cs_error.append(np.linalg.norm(x-x_mean))

    return gradient_history ,v_history, loss_history, accuracy_history,cs_error,x

def stable_inverse_diag(W):
    epsilon=0
    W_diag = np.diag(W).copy()
    W_diag[W_diag < epsilon] = epsilon 
    return np.diag(1/W_diag)


def pull_sum_real_data(A, init_x, h_data, y_data, X_test, y_test, grad_func, loss_func, grad_f_bar_func, accuracy_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    """ 实际上是pull diag算法 """
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    epsilon=1e-15
    w = stable_inverse_diag(W)@g 
    v= g
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    gradient_history=[np.linalg.norm(gradient_f_bar_x_val)]
    Ls=[1/min(np.diag(A))]
    v_history=[np.linalg.norm(v)]
    loss_history=[loss_func(np.dot(pi,x), y=y, h=h, rho=rho)]
    accuracy_history=[accuracy_func(A0@x, X_test, y_test)]
    x_mean=np.mean(x,axis=0)
    cs_error=[np.linalg.norm(x-x_mean)]

    for i in range(max_it):
        W=A@W
        x = A @ x - lr *  v
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = A @ v + stable_inverse_diag(W)@g - w
        w = stable_inverse_diag(W)@g#这一步计算的w是下一步用到的w，因此程序没有问题
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1))
        Ls.append(1/min(np.diag(W)))
        v_history.append(np.linalg.norm(v))
        loss_history.append(loss_func(np.dot(pi,x), y=y, h=h, rho=rho))
        accuracy_history.append(accuracy_func(A0@x, X_test, y_test))
        x_mean=np.mean(x,axis=0)
        cs_error.append(np.linalg.norm(x-x_mean))

    return gradient_history, v_history, max(Ls), loss_history, accuracy_history,cs_error,x