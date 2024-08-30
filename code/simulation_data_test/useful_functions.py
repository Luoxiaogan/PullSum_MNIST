import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import networkx as nx
from mpmath import mp

#---------------------------------------------------------------------------

#获得矩阵的右perron向量
def get_right_perron(W):
    """ 对于列随机矩阵，获得矩阵的右perron向量 """
    c = np.linalg.eig(W) 
    eigenvalues = c[0]#特征值，向量
    eigenvectors = c[1]#特征向量，矩阵
    max_eigen = np.abs(eigenvalues).argmax()#返回绝对值最大的特征值对应的位置
    vector = c[1][:,max_eigen]#max_eigen那一列
    return np.abs(vector / np.sum(vector))#单位化

#获得矩阵的左perron向量
def get_left_perron(W):
    """ 对于行随机矩阵，获得矩阵的左perron向量 """
    return get_right_perron(W.T)#计算转置的右perron即可

#---------------------------------------------------------------------------

def compute_kappa_row(A):
    pi=get_left_perron(A)
    return np.max(pi)/np.min(pi)

def compute_kappa_col(B):
    pi=get_right_perron(B)
    return np.max(pi)/np.min(pi)

#计算第二大特征值的模长
def compute_2st_eig_value(A):
    return abs(np.linalg.eigvals(A)[1])

def compute_beta_row(A, precision=64):
    mp.dps = precision  # 设置计算精度
    n = A.shape[0]
    pi = get_left_perron(A)
    one = np.ones(n)
    if not nx.is_strongly_connected(nx.DiGraph(A)):
        print("不是强联通")
    matrix = A - np.outer(one, pi)
    diag1 = np.diag(np.sqrt(pi))
    diag1_inverse = np.diag(1 / np.sqrt(pi))
    result = np.linalg.norm(diag1 @ matrix @ diag1_inverse, 2)
    return min(result, 1)  # 裁剪结果不超过1

def compute_beta_col(B, precision=64):
    mp.dps = precision  # 设置计算精度
    n = B.shape[0]
    pi = get_right_perron(B)
    one = np.ones(n)
    if not nx.is_strongly_connected(nx.DiGraph(B)):
        print("不是强联通")
    matrix = B - np.outer(pi, one)
    diag1 = np.diag(np.sqrt(pi))
    diag1_inverse = np.diag(1 / np.sqrt(pi))
    result = np.linalg.norm(diag1_inverse @ matrix @ diag1, 2)
    return min(result, 1)  # 裁剪结果不超过1

def compute_S_A_row(A):
    kappa=compute_kappa_row(A)
    beta=compute_beta_row(A)
    n=A.shape[0]
    output=2*np.sqrt(n)*(1+np.log(kappa))/(1-beta)
    return output

def compute_S_B_col(B):
    kappa=compute_kappa_col(B)
    beta=compute_beta_col(B)
    n=B.shape[0]
    output=2*np.sqrt(n)*(1+np.log(kappa))/(1-beta)
    return output

def show_row(A):
    print("A的第二大特征值:",compute_2st_eig_value(A))
    print("A的beta:",compute_beta_row(A))
    print("A的spectral gap:",1-compute_beta_row(A))
    print("A的kappa:",compute_kappa_row(A))
    print("S_A是:",compute_S_A_row(A),"\n")

def show_col(B):
    print("B的第二大特征值:",compute_2st_eig_value(B))
    print("B的beta:",compute_beta_col(B))
    print("B的spectral gap:",1-compute_beta_col(B))
    print("B的kappa:",compute_kappa_col(B))
    print("S_B是:",compute_S_B_col(B),"\n")

#---------------------------------------------------------------------------

def stable_log_exp(x):
    """Stable computation of log(1 + exp(x)) to avoid overflow."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def loss(x, y, h, rho=0.001):
    n, L, d = h.shape
    x = x.reshape(-1)
    
    # 计算 h_dot_x
    h_dot_x = np.einsum('ijk,k->ij', h, x)  # (n, L)
    
    # 计算 stable_log_exp
    term1 = np.sum(stable_log_exp(-y * h_dot_x)) / L  # 标量
    
    # 计算正则化项
    term2 = np.sum(rho * x**2 / (1 + x**2))  # 标量
    
    return term1 + term2

def grad(x, y, h, rho=0.001):
    n, L, d = h.shape
    
    # 计算 h_dot_x
    h_dot_x = np.einsum('ijk,ik->ij', h, x)  # (n, L)
    
    # 计算 exp_val 并处理溢出
    exp_val = np.exp(y * h_dot_x)
    np.clip(exp_val, None, 1e300, out=exp_val)  # 防止数值溢出
    
    # 计算 g1
    g1 = -np.einsum('ijk,ij->ik', h, y / (1 + exp_val)) / L  # (n, d)
    
    # 计算 g2
    g2 = 2 * x / (1 + x ** 2) ** 2  # (n, d)
    
    # 返回最终的梯度
    return (g1 + g2 * rho).reshape(n, d)

def grad_f_bar_x(x, y, h, rho=0.001):
    output=grad(x=x,y=y,h=h,rho=rho)
    n=x.shape[0]
    return np.sum(output,axis=0)/n

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

#---------------------------------------------------------------------------

def init_data(n=6,d=5,L=200,seed=42,sigma_h=10):
    """ 初始数据 """
    np.random.seed(seed)
    x_opt=np.random.normal(size=(1,d))#生成标准正态的向量,是最优解
    x_star=x_opt+sigma_h*np.random.normal(size=(n,d))#后面的一项可以理解成噪声项
    h=np.random.normal(size=(n, L, d))
    y=np.zeros((n,L))
    for i in range(n):
        for l in range(L):
            z=np.random.uniform(0, 1)
            if 1/z > 1 + np.exp(-np.inner(h[i,l,:],x_star[i])):
                y[i,l]=1
            else:
                y[i,l]=-1
    return (h,y,x_opt,x_star)

def init_x_func(n=6, d=10, seed=42):
    np.random.seed(seed)
    return 0.01 * np.random.normal(size=(n, d))

#---------------------------------------------------------------------------

def generate_column_stochastic_matrix(n, seed_location=42, seed_value=43, seed_num=44):
    """
    生成一个列随机矩阵，每列的元素之和为1。该矩阵的每列具有随机数量的非零元素，
    其数量在1到n之间，其中n为矩阵的维度。

    Parameters:
    n (int): 矩阵的维度，即矩阵是 n x n 的。
    seed1 (int): 用于确定每列非零元素的位置的随机种子。
    seed2 (int): 用于赋予每列非零元素随机值的随机种子。
    seed3 (int): 用于确定每列非零元素的数量的随机种子。

    Returns:
    np.ndarray: 一个 n x n 的列随机矩阵，其中每列的元素和为1。

    注意:
    - 矩阵中的元素值是随机分配的，但每列的总和被标准化为1。
    """
    seed1=seed_location
    seed2=seed_value
    seed3=seed_num

    np.random.seed(seed3)
    k_values = np.random.randint(1, n+1, size=n)  # 每列非零元素的数量介于1到n之间

    M = np.zeros((n, n))
    nonzero_positions = []
    np.random.seed(seed1)
    for j, k in enumerate(k_values):
        indices = np.random.choice(n, k, replace=False)
        nonzero_positions.append(indices)

    np.random.seed(seed2)
    for j, indices in enumerate(nonzero_positions):
        M[indices, j] = np.random.rand(len(indices))

    column_sums = np.sum(M, axis=0)
    M[:, column_sums > 0] /= column_sums[column_sums > 0]

    return M

def column_to_row_stochastic(B, seed=None):
    """
    将给定的列随机矩阵转换为行随机矩阵，同时保持与原矩阵相同的网络结构。
    这意味着转换后的矩阵将在相同的位置具有非零元素，从而保持节点间的传递关系不变。
    转换过程通过随机分配新的行随机值来实现，同时确保每一行的元素和为1。

    Parameters:
    B (np.ndarray): 输入的列随机矩阵，假定每一列的和为1。
    seed (int, optional): 可选的随机种子，用于确保随机值的可重复性。

    Returns:
    np.ndarray: 转换后的行随机矩阵，其中每一行的和为1。

    注意:
    - 输入矩阵 B 的每一行不一定需要有非零元素，但转换过程确保至少每行将有一些非零值。
    - 若行完全由零组成，则该行保持不变（全零行）。
    - 保持原始矩阵的网络结构不变是此转换的一个重要特点，确保了节点间的连接关系在转换过程中不会改变。
    """
    n = B.shape[0]
    A = np.zeros_like(B)
    
    if seed is not None:
        np.random.seed(seed)  # 设置随机种子以保证可重复性

    for i in range(n):
        # 找到B中第i行非零的列索引
        nonzero_indices = np.nonzero(B[i, :])[0]
        # 生成随机值并赋给这些非零位置
        random_values = np.random.rand(len(nonzero_indices))
        # 标准化随机值使得这一行的和为1
        A[i, nonzero_indices] = random_values / random_values.sum()
    
    return A

def generate_row_stochastic_matrix(n, seed_location=42, seed_value=43, seed_num=44):
    """
    生成一个行随机矩阵，每行的元素之和为1。该矩阵的每行具有随机数量的非零元素，
    其数量在1到n之间，其中n为矩阵的维度。

    Parameters:
    n (int): 矩阵的维度，即矩阵是 n x n 的。
    seed1 (int): 用于确定每行非零元素的位置的随机种子。
    seed2 (int): 用于赋予每行非零元素随机值的随机种子。
    seed3 (int): 用于确定每行非零元素的数量的随机种子。

    Returns:
    np.ndarray: 一个 n x n 的行随机矩阵，其中每行的元素和为1。

    注意:
    - 矩阵中的元素值是随机分配的，但每行的总和被标准化为1（对于非全零行）。
    """
    seed1=seed_location
    seed2=seed_value
    seed3=seed_num

    np.random.seed(seed3)
    k_values = np.random.randint(1, n+1, size=n)  # 每行非零元素的数量介于1到n之间

    M = np.zeros((n, n))
    nonzero_positions = []
    np.random.seed(seed1)
    for i, k in enumerate(k_values):
        indices = np.random.choice(n, k, replace=False)
        nonzero_positions.append(indices)

    np.random.seed(seed2)
    for i, indices in enumerate(nonzero_positions):
        M[i, indices] = np.random.rand(len(indices))

    # 标准化每行以使其总和为1
    row_sums = np.sum(M, axis=1)
    M[row_sums > 0] = (M[row_sums > 0].T / row_sums[row_sums > 0]).T

    return M

def row_to_column_stochastic(A, seed=None):
    """
    将给定的行随机矩阵转换为列随机矩阵，同时保持与原矩阵相同的网络结构。
    这意味着转换后的矩阵将在相同的位置具有非零元素，从而保持节点间的传递关系不变。
    转换过程通过随机分配新的列随机值来实现，同时确保每一列的元素和为1。

    Parameters:
    A (np.ndarray): 输入的行随机矩阵，假定每一行的和为1。
    seed (int, optional): 可选的随机种子，用于确保随机值的可重复性。

    Returns:
    np.ndarray: 转换后的列随机矩阵，其中每一列的和为1。

    注意:
    - 输入矩阵 A 的每一列不一定需要有非零元素，但转换过程确保至少每列将有一些非零值。
    - 若列完全由零组成，则该列保持不变（全零列）。
    - 保持原始矩阵的网络结构不变是此转换的一个重要特点，确保了节点间的连接关系在转换过程中不会改变。
    """
    n = A.shape[0]
    B = np.zeros_like(A)
    
    if seed is not None:
        np.random.seed(seed)  # 设置随机种子以保证可重复性

    for j in range(n):
        # 找到A中第j列非零的行索引
        nonzero_indices = np.nonzero(A[:, j])[0]
        # 生成随机值并赋给这些非零位置
        random_values = np.random.rand(len(nonzero_indices))
        # 标准化随机值使得这一列的和为1
        B[nonzero_indices, j] = random_values / random_values.sum()
    
    return B

#---------------------------------------------------------------------------
def random_process(mat,p=0.1,r=100,seed=42):    # 概率p和除数r
    np.random.seed(seed)
    # 生成布尔型数组
    mask = np.random.choice([True, False], size=mat.shape[0], p=[p, 1-p])

    # 将布尔型数组转化为浮点型数组
    divisor = np.ones_like(mask, dtype=float)
    divisor[mask] = 1/r

    # 将除数数组与原始矩阵相乘
    mat = mat * divisor[:, np.newaxis]

    # 重新归一化每一列元素 (使用每列元素之和)
    col_sums = np.sum(mat, axis=0)
    mat /= col_sums.reshape((1, -1))  
    return mat

def get_xinmeng_like_matrix(n,seed=42):
    np.random.seed(seed)
    M = np.zeros((n, n))

    # 主对角线上的元素
    M[np.diag_indices(n)] = np.random.rand(n)

    # 次对角线上的元素
    for i in range(n-1):
        M[i+1, i] = np.random.rand()

    # 第一行上的元素
    M[0, :] = np.random.rand(n)
    M /= np.sum(M,axis=0)
    return M

def get_xinmeng_matrix(n):
    M = np.zeros((n, n))

    # 主对角线上的元素
    M[np.diag_indices(n)] = 1/3*np.ones(n)
    M[n-1,n-1]=M[n-1,n-1]+1/3
    
    # 次对角线上的元素
    for i in range(n-1):
        M[i+1, i] = M[i+1,i]+1/3
    
    # 第一行上的元素
    M[0, :] = M[0,:]+1/3
    
    return M


#指定n,快速生成列随机矩阵

def get_mat1(n):
    W = np.random.rand(n,n)
    col_sum = np.sum(W,axis=0)
    return W / col_sum

def get_bad_mat(n=30,p=0.1,show_graph=0,seed=42,verbose=1):
    # 生成稀疏随机矩阵，保证强连通
    M = np.zeros((n, n))
    cnt=0
    np.random.seed(seed)
    while not nx.is_strongly_connected(nx.DiGraph(M)):
        M = np.random.choice([0, 1], size=(n, n), p=[1-p, p])
        cnt=cnt+1
        if cnt>1000000:
            raise Exception("1000000次都没找到合适的矩阵")
    if verbose==1:
        print('用了'+str(cnt)+'次找到')
    # 归一化每列元素，使得每列元素之和为1
    col_sums = np.sum(M, axis=0)
    M = M / col_sums

    # 将矩阵转换成有向图，并绘制出该图
    if show_graph==1:
        G = nx.DiGraph(M)
        nx.draw(G, with_labels=True)
        plt.show()
        diameter = nx.algorithms.distance_measures.diameter(G)
        print(f"图的直径为{diameter}")
    return M

def process(M, r=100):
    while True:
        # 计算每行元素之和
        row_sums = np.sum(M, axis=1)

        # 找到最小行和以及对应的行索引
        row_min = np.argmin(row_sums)
        s_min = row_sums[row_min]

        # 如果最大行和与最小行和的比值已经满足要求，则跳出循环
        if np.max(row_sums) / s_min >= r:
            break
        # 将最小行除以 ratio并归一化
        M[row_min] /= r 
        col_sums = np.sum(M, axis=0)
        M /= col_sums.reshape((1, -1))
    
    # 重新归一化每一列元素 (使用每列元素之和)
    col_sums = np.sum(M, axis=0)
    M /= col_sums.reshape((1, -1))

def ring(n):
    M=np.eye(n)
    for i in range(n-1):
        M[i+1,i]=1
        M[i,i+1]=1
    return M
def grid(n):
    # 创建一个n*n的grid graph  
    G = nx.grid_2d_graph(n, n)  
    # 获取节点的排列  
    nodes = list(G.nodes)  
    # 生成邻接矩阵  
    adj_matrix = nx.adjacency_matrix(G)  
    # 将稀疏矩阵转换为numpy数组  
    adj_matrix = adj_matrix.toarray()  
    return adj_matrix*0.5+0.5*np.eye(n*n)

def Row(matrix):  
    # 计算每一行的和  
    M=matrix.copy()
    row_sums = np.sum(M, axis=1)  
  
    # 将每一行除以该行的和  
    for i in range(M.shape[0]):  
        M[i, :] /= row_sums[i]  
  
    return M 

def Col(matrix):  
    W=matrix.copy()
    # 计算每一行的列
    col_sums = np.sum(W, axis=0)  
  
    # 将每一列除以该行的和  
    for i in range(W.shape[0]):  
        W[:, i] /= col_sums[i]  
  
    return W

def get_B(A,u,n):
    v=np.ones(n)
    for _ in range(u):
        v=A.T@v
    v1=A.T@v
    return np.diag(v)@A@np.diag(1/v1)

#---------------------------------------------------------------------------

def prettyshow(grads,legend,save='image.pdf',ylabel='Gradient Norm'):
    # plot the results
    plt.rcParams['figure.figsize'] = 5, 4
    plt.figure()
    xlen = len(grads[0])
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan']
    markers = ['d', '^', 'o', '<', '*', 's']
    idx_set = np.arange(0, xlen, xlen//10)
    for i in range(len(grads)):
        plt.semilogy(0, grads[i][0], color=colors[i], marker=markers[i], markersize = 7)
    for i in range(len(grads)):
        for idx in idx_set:
            plt.semilogy(idx, grads[i][idx], color=colors[i], marker=markers[i], markersize = 7, linestyle = 'None')
    for i in range(len(grads)):
        plt.semilogy(np.arange(xlen), grads[i], linewidth=1.0, color=colors[i])
    plt.legend(legend, fontsize=12)
    plt.xlabel('Iteration', fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.grid(True)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(save)
    plt.show()