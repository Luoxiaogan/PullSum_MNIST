import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from new_function import *

def get_B(A,u,n):
    v=np.ones(n)
    for _ in range(u):
        v=A.T@v
    v1=A.T@v
    return np.diag(v)@A@np.diag(1/v1)

def new_pull_sum(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    #v = np.ones((n, d)) #修改1
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g #修改2
    B=get_B(A,9,n)
    correction_vec=np.ones(n) #修改3
    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n

    for i in range(max_it):
        
        correction_vec=A.T@correction_vec #修改4
        x = A @ x - lr * np.diag(1/correction_vec) @ v #修改5
        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
    

    return max_it, history_gradient_two_norm, cs_error, loss_history, x,B

def pull_sum(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    w = np.linalg.inv(np.diag(np.diag(W)))@g
    v= g
    

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n

    for i in range(max_it):

        W=A@W
        
        x = A @ x - lr *  v
        

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = A @ v + np.linalg.inv(np.diag(np.diag(W))+10**(-i)*np.eye(n))@g - w
        w = np.linalg.inv(np.diag(np.diag(W))+10**(-i)*np.eye(n))@g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
    

    return max_it, history_gradient_two_norm, cs_error, loss_history, x


def push_pull(A, B, init_x, h_data, y_data, X_test, y_test, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    v = np.ones((n, d))
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):


        x = A @ x - lr * v
        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, accuracy, x

def push_pull_no_accuracy(A, B, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    v = np.ones((n, d))
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):


        x = A @ x - lr * v
        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, x

def push_pull_change_no_accuracy(A, B, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    v = np.ones((n, d))
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g
    W = np.eye(n)

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):

        W=B@W
        vector=W@np.ones(n)
        D_inverse=np.diag(1/vector)

        x = A @ x - lr * D_inverse @ v
        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, x

def push_diging(C, init_x, h_data, y_data,X_test, y_test, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1, sigma_n=0.1,max_it=200, mg=1, decay=1):
    n = C.shape[1]
    h, y = h_data, y_data
    x = init_x
    u = x.copy()
    v = np.ones(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    t = g

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #x_history = np.zeros((max_it + 1, *x.shape))
    accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)

    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)

    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #x_history[0] = x.copy()
    accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):
        u = np.dot(C, (u - lr * t))
        v = np.dot(C, v)
        x = u / v[:, None]#x = np.diag(1 / v) @ u

        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        t = np.dot(C, t) + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次
        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)

        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)

        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #x_history[i + 1] = x.copy()
        accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, accuracy,x#, x_history


def push_diging(C, init_x, h_data, y_data,X_test, y_test, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1, sigma_n=0.1,max_it=200, mg=1, decay=1):
    n = C.shape[1]
    h, y = h_data, y_data
    x = init_x
    u = x.copy()
    v = np.ones(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    t = g

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #x_history = np.zeros((max_it + 1, *x.shape))
    accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)

    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)

    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #x_history[0] = x.copy()
    accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):
        u = np.dot(C, (u - lr * t))
        v = np.dot(C, v)
        x = u / v[:, None]#x = np.diag(1 / v) @ u

        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        t = np.dot(C, t) + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次
        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)

        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)

        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #x_history[i + 1] = x.copy()
        accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, accuracy,x#, x_history

def push_diging_no_accuracy(C, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1, sigma_n=0.1,max_it=200, mg=1, decay=1):
    n = C.shape[1]
    h, y = h_data, y_data
    x = init_x
    u = x.copy()
    v = np.ones(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    t = g

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #x_history = np.zeros((max_it + 1, *x.shape))
    #accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)

    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)

    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #x_history[0] = x.copy()
    #accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):
        u = np.dot(C, (u - lr * t))
        v = np.dot(C, v)
        x = u / v[:, None]#x = np.diag(1 / v) @ u

        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        t = np.dot(C, t) + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次
        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)

        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)

        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #x_history[i + 1] = x.copy()
        #accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history,x#, x_history

def push_pull_change_no_accuracy(A, B, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    v = np.ones((n, d))
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g
    W = np.eye(n)

    history_gradient_two_norm = np.zeros(max_it + 1)
    cs_error = np.zeros(max_it + 1)
    loss_history = np.zeros(max_it + 1)
    #accuracy = np.zeros(max_it + 1)

    gradient_f_bar_x_val = grad_f_bar_func(x, y, h, rho=rho)
    history_gradient_two_norm[0] = np.linalg.norm(gradient_f_bar_x_val, 2)
    x_mean = np.sum(x, axis=0) / n
    cs_error[0] = np.linalg.norm(x - x_mean, 2)
    loss_history[0] = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均
    #accuracy[0] = compute_accuracy(x, X_test, y_test)

    for i in range(max_it):

        W=B@W
        vector=W@np.ones(n)
        D_inverse=np.diag(1/vector)

        x = A @ x - lr * D_inverse @ v
        pre_g = g

        gradient=grad_func(x, y, h, rho=rho)

        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g

        gradient_f_bar_x_val = np.mean(gradient, axis=0)#只用算一次

        norm1 = np.linalg.norm(gradient_f_bar_x_val, 2)
        x_mean = np.sum(x, axis=0) / n
        norm2 = np.linalg.norm(x - x_mean, 2)
        loss1 = np.sum(np.array([loss_func(np.ascontiguousarray(x[k]), y, h, rho=rho) for k in range(n)]))/n#修改了，这里是求和，而不是求平均

        history_gradient_two_norm[i + 1] = norm1
        cs_error[i + 1] = norm2
        loss_history[i + 1] = loss1
        #accuracy[i + 1] = compute_accuracy(x, X_test, y_test)

    return max_it, history_gradient_two_norm, cs_error, loss_history, x