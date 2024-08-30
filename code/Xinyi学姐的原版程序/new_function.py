import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import networkx as nx

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

#获得矩阵2-范数
def tow_norm_matrix(A):
    """ 获得矩阵2-范数 """
    return np.sqrt(np.max(np.abs(np.linalg.eig(A.T @ A)[0])))

#获得向量的2-范数
def tow_norm_vector(v):
    return np.sqrt(sum(i**2 for i in v))

#获得矩阵第二大特征值
def get_second_eigenvalue(A):
    """ 获得矩阵第二大特征值 """
    return sorted(np.abs(np.linalg.eig(A)[0]))[-2]

#给定向量和left_perron，返回pi_l范数
def pi_l_norm_vector(v,pi_l):
    sqrt_pi_l=np.sqrt(np.diag(pi_l))
    product=np.dot(sqrt_pi_l, v)
    return np.linalg.norm(product, ord=2)

#给定向量和right_perron，返回pi_r范数
def pi_r_norm_vector(v,pi_r):
    sqrt_inv_pi_r=np.sqrt(np.diag(1 / pi_r))
    product=np.dot(sqrt_inv_pi_r, v)
    return np.linalg.norm(product, ord=2)

#给定矩阵和left_perron，返回pi_l范数
def pi_l_norm_matrix(M,pi_l):
    """ 这里要求方阵 """
    diag1=np.diag(np.sqrt(pi_l))
    diag1_inverse=np.diag(1/np.sqrt(pi_l))
    return np.linalg.norm(diag1@M@diag1_inverse,ord=2)

#给定矩阵和right_perron，返回pi_r范数
def pi_r_norm_matrix(M,pi_r):
    """ 这里要求方阵 """
    diag2=np.diag(np.sqrt(pi_r))
    diag2_inverse=np.diag(1/np.sqrt(pi_r))
    return np.linalg.norm(diag2_inverse@M@diag2,ord=2)

#定义损失函数
def loss(x,y,h,rho=0.001):#x: d*1 y: n*L, h: n*L*d, g:n*d
    """ 损失函数,计算的是一个标量 """
    n,L,d=h.shape
    x=x.reshape(-1) #转化成了行向量
    result=0
    for i in range(n):#n个节点
        result += sum([np.log(1+np.exp(-y[i,l]*np.inner(h[i,l,:],x))) for l in range(L)]) / L
        #每一个节点里面的Logistic损失函数，y[i,l]，h[i,l,:]对应第i个节点的第l个分量
    result += rho * sum([x[j]**2/(1+x[j]**2) for j in range(d)])#正则化参数
    return result

def grad(x, y, h, rho=0.001):
    """
    计算局部参数的梯度。

    参数：
    x: numpy array, 形状为 (n, d)，输入参数矩阵，其中 n 是样本数量，d 是参数的维度。
    y: numpy array, 形状为 (n, L)，标签矩阵，其中 L 是每个样本的标签数量。
    h: numpy array, 形状为 (n, L, d)，参数矩阵，其中 n 是样本数量，L 是每个样本的标签数量，d 是参数的维度。这里的 h 可以看作是输入参数的一种表示形式。
    rho: float, 正则化参数，默认值为 0.001。

    返回值：
    numpy array, 形状为 (n, d)，每一行是局部参数 x_i 求的局部损失函数的梯度 ∇f_i(x_i)。
    """

    n, L, d = h.shape
    g1 = np.zeros((n, d))
    g2 = np.zeros((n, d))

    # 计算损失函数关于数据项的梯度
    for i in range(n):
        for j in range(d):
            g1[i, j] = sum([-y[i, l] * h[i, l, j] / (1 + np.exp(y[i, l] * np.inner(h[i, l, :], x[i, :]))) for l in range(L)]) / L

    # 计算正则项的梯度
    for i in range(n):
        for j in range(d):
            g2[i, j] = 2 * x[i, j] / (1 + x[i, j] ** 2) ** 2

    # 将两部分的梯度加权求和，并乘以正则化参数
    return g1 + g2 * rho


def grad_f_bar_x(x, y, h, rho=0.001):
    output=grad(x=x,y=y,h=h,rho=rho)
    n=x.shape[0]
    return np.sum(output,axis=0)/n


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




def push_pull(A,B,data,d=5,L=200,rho=0.1,lr=0.1,sigma_n=0,max_it=200,mg=1,seed=42,decay=1):
    """ 这里的sigma_n 实际上是没有用到的，因为这里没有考虑随机梯度 """
    np.random.seed(seed)
    n=A.shape[0]#A的行数就是节点的数目n
    h,y,x_opt,x_star=data
    x=init_x_func(n=n,d=d)#初始的x，有n个节点
    w=x#备份
    lr0=lr#初始化学习率
    v=np.ones(n)#全部都是1的向量
    g=grad(w,y,h,rho=rho).reshape(x.shape)#初始化梯度
    v=g#初始化gradient tracking

    #求出perron_vector
    pi_l=get_left_perron(A)
    pi_r=get_right_perron(B)

    #记录
    history_x,history_gradient,history_gradient_tow_norm,history_gradient_pi_l_norm,history_gradient_pi_r_norm,cs_error,loss_history=[],[],[],[],[],[],[]

    history_x.append(x)#记录x的取值
    #每一个梯度g实际上是一个n*d的矩阵，总的f的梯度是每一行的梯度的求和在除以n，之后再对它求范数
    gradient_f=np.sum(g,axis=1)/n
    history_gradient.append(gradient_f)#记录梯度的取值
    history_gradient_tow_norm.append(tow_norm_vector(gradient_f))#tow_norm
    history_gradient_pi_l_norm.append(pi_l_norm_vector(v=gradient_f,pi_l=pi_l))#pi_l_norm
    history_gradient_pi_r_norm.append(pi_r_norm_vector(v=gradient_f,pi_r=pi_r))#pi_r_norm
    cs_error.append(np.linalg.norm(x-np.mean(x,axis=0))) #consensus error
    loss_history.append(sum([loss(x[k],y,h,rho=rho) for k in range(n)])) #loss_history
    for i in range(max_it):
        
        x=A@x-lr*v
        pre_g=g
        g=grad(x,y,h,rho=rho).reshape(x.shape)
        v=B@v+g-pre_g
        
        gradient_f=np.sum(g,axis=1)/n
        history_gradient.append(gradient_f)#记录梯度的取值

        history_x.append(x)
        history_gradient.append(g)
        history_gradient_tow_norm.append(tow_norm_vector(gradient_f))
        history_gradient_pi_l_norm.append(pi_l_norm_vector(v=gradient_f,pi_l=pi_l))#pi_l_norm
        history_gradient_pi_r_norm.append(pi_r_norm_vector(v=gradient_f,pi_r=pi_r))#pi_r_norm
        cs_error.append(np.linalg.norm(x-np.mean(x,axis=0))) 
        loss_history.append(sum([loss(x[k],y,h,rho=rho) for k in range(n)])) #loss_history
        
    result_table = pd.DataFrame({
        "Iteration": range(1+max_it),
        "Gradient_tow_norm": history_gradient_tow_norm,
        "Gradient_pi_l_norm": history_gradient_pi_l_norm,
        "Gradient_pi_r_norm": history_gradient_pi_r_norm,
        "Consensus_error":cs_error,
        "Loss":loss_history,
    })
    
    return result_table


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