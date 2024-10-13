import torch
import torch.nn as nn
import networkx as nx
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np

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

def get_B(A,u,n):
    v=np.ones(n)
    for _ in range(u):
        v=A.T@v
    v1=A.T@v
    return np.diag(v)@A@np.diag(1/v1)

#指定n,快速生成列随机矩阵

def get_mat1(n):
    W = np.random.rand(n,n)
    col_sum = np.sum(W,axis=0)
    return W / col_sum

def get_bad_mat(n=10,p=0.15,show_graph=0,seed=42,verbose=1):
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

def get_B(A,u,n):
    v=np.ones(n)
    for _ in range(u):
        v=A.T@v
    v1=A.T@v
    return np.diag(v)@A@np.diag(1/v1)

def test_row(A,epochs=100,if_plot=False):
    # For Pull Sum and Pull Diag
    n=A.shape[0]
    c=np.ones(n)
    W=np.eye(n)
    list1,list2=[],[]
    for i in range(epochs):
        c=A.T@c
        W=A@W
        list1.append((1/(min(c))))
        list2.append(1/min(np.diag(W)))
    if if_plot:
        plt.plot(list1,color='r',label='Sum, 1/min(correction_vector)')
        plt.plot(list2,color='b',label='Diag, 1/min(Diag(W))')
        plt.title('Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('1/min')
        plt.legend() 
        plt.show()

        plt.plot(list1,color='r',label='Sum, 1/min(correction_vector)')
        plt.title('Sum, 1/min(correction_vector)')
        plt.xlabel('Epoch')
        plt.ylabel('1/min')
        plt.legend() 
        plt.show()
    return max(list1),max(list2)

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