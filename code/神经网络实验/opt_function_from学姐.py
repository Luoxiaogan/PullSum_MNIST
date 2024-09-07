import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from useful_functions import *

def new_pull_sum_1(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g 
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    
    correction_vec=np.ones(n)
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    gradient_history=[np.linalg.norm(gradient_f_bar_x_val)]
    v_history=[np.linalg.norm(v)]
    for i in range(max_it):
        pre_vec=correction_vec
        correction_vec=A.T@correction_vec 
        x = A @ x - lr * np.diag(1/correction_vec) @ v 
        pre_g = g
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        B=np.diag(pre_vec)@A@np.diag(1/correction_vec)
        v = B @ v + g - pre_g
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1))
        v_history.append(np.linalg.norm(np.diag(1/correction_vec) @v))

    return gradient_history 

def pull_sum_1(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    """ 实际上是pull diag算法 """
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    w = np.linalg.inv(np.diag(np.diag(W)))@g
    v= g
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    g0=np.linalg.norm(gradient_f_bar_x_val)
    gradient_history=[1]
    Ls=[1/min(np.diag(A))]
    v_history=[np.linalg.norm(v)]
    for i in range(max_it):
        W=A@W
        x = A @ x - lr *  v
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = A @ v + np.linalg.inv(np.diag(np.diag(W)))@g - w
        w = np.linalg.inv(np.diag(np.diag(W)))@g#这一步计算的w是下一步用到的w，因此程序没有问题
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1)/g0)
        Ls.append(1/min(np.diag(W)))
        v_history.append(np.linalg.norm(v))

    return gradient_history

def get_B(A,u,n):
    v=np.ones(n)
    for _ in range(u):
        v=A.T@v
    v1=A.T@v
    return np.diag(v)@A@np.diag(1/v1)

def new_pull_sum_11(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    v = g 
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    B=get_B(A,2*n,n)
    correction_vec=np.ones(n)
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    g0=np.linalg.norm(gradient_f_bar_x_val)
    gradient_history=[1]
    v_history=[np.linalg.norm(v)]
    for i in range(max_it):
        pre_vec=correction_vec
        correction_vec=A.T@correction_vec 
        x = A @ x - lr * np.diag(1/correction_vec) @ v 
        pre_g = g
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = B @ v + g - pre_g
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1)/g0)
        v_history.append(np.linalg.norm(np.diag(1/correction_vec) @v))

    return gradient_history 


def FRSD(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1, beta=0.1 ,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    w = np.linalg.inv(np.diag(np.diag(W)))@g
    v= g
    u=np.zeros((n, d))
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    g0=np.linalg.norm(gradient_f_bar_x_val)
    gradient_history=[1]
    for i in range(max_it):
        z=A@x
        if i!=0:
            u=u+beta*(x-z)
        g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v=np.linalg.inv(np.diag(np.diag(W)))@g
        x=z-lr*(v+u)
        W=W@A
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1)/g0)
    return gradient_history
    


def FROZEN(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1 , beta=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    w = np.linalg.inv(np.diag(np.diag(W)))@g
    v= g
    s=np.zeros((n, d))
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    g0=np.linalg.norm(gradient_f_bar_x_val)
    gradient_history=[1]
    for i in range(max_it):
        s_pre=s
        s=A@x-lr*v
        x=s+beta*(s-s_pre)
        pre_g = g
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v = A @ v + np.linalg.inv(np.diag(np.diag(W@A)))@g - np.linalg.inv(np.diag(np.diag(W)))@pre_g
        W=W@A
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1)/g0)
    return gradient_history
    
def diag_sgd(A, init_x, h_data, y_data, grad_func, loss_func, grad_f_bar_func, d=784, L=1, rho=0.1, lr=0.1,sigma_n=0.1, max_it=200, mg=1, decay=1):
    """ 实际上是pull diag算法 """
    n = A.shape[0]
    h, y = h_data, y_data
    x = init_x
    W = np.eye(n)
    g = grad_func(x, y, h, rho=rho).reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
    w = np.linalg.inv(np.diag(np.diag(W)))@g
    v= g
    pi=get_left_perron(A).reshape((1,n))
    A0=np.ones((n, 1))@pi@A
    gradient_f_bar_x_val = grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
    g0=np.linalg.norm(gradient_f_bar_x_val)
    gradient_history=[1]
    for i in range(max_it):
        
        gradient=grad_func(x, y, h, rho=rho)
        g = gradient.reshape(x.shape)+sigma_n/mg*np.random.normal(size=(n,d))
        v=np.linalg.inv(np.diag(np.diag(W)))@g
        x=A@x-lr*v
        W=W@A
        g1=grad_f_bar_func((A0@x).reshape(x.shape), y, h, rho=rho)
        gradient_history.append(np.linalg.norm(g1)/g0)
    return gradient_history
