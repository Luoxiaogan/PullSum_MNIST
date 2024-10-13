import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# @Lg 难点在于训练时间和跑收敛。建议现在先把MNIST的跑起来，然后写CIFAR的代码。如果Cifar-10 实验在10个节点无法收敛，可以考虑构造5个节点的稀疏图比如di_ring(n=5)和 Row(get_xinmeng_matrix(n=5))
# 一、 4层神经网络训练MNIST数据集：
# 【1】异质性观察
# 图1：pulldiag在di_ring(n=5)+三种不同异质性的表现(注：现在只弄了两种异质性，均匀分布和完全异质分布。能不能弄一个稍微混合一点的数据分布？比如在完全异质分布条件下，让1，2号节点数据混合一下。）
# 图2：pullsum在di_ring(n=5)+三种不同异质性的表现
# 图3：pulldiag和pullsum都在di_ring(n=5)或者di_ring(n=10)+强异质性下的对比表现

# 【2】拓扑影响
# 图1：在row_and_col_mat(n=10, p=0.5）+强异质性条件下比较pulldiag, pullsum, frsd, frozen 
# 图2：在row_and_col_mat(n=10, p=0.2）+强异质性条件下比较pulldiag, pullsum, frsd, frozen 
# 图3：只看pullsum, 在row_and_col_mat(n=10, p=0.5），row_and_col_mat(n=10, p=0.2），di_ring(n=10)，grid_10()上的表现。


# 二、Resnet训练CIFAR-10数据

# 图1：di_ring(n=10)+强异质性条件下比较 pulldiag, pullsum, frsd, frozen
# 图2：grid_10()+强异质性条件下比较 pulldiag, pullsum, frsd, frozen



# @cxy: 合成数据的实验难点在于怎么让强异质性真正发挥作用。我的初步建议是提升问题的维度，比如令d=20，在此基础上提高sigma_h。如果不行，要等吴百濠学长的异质性代码。
# 一、【确定性情况+强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 二、【噪声取1e-3, 1e-2, 1e-1 ，强异质性。对比pullsum 和其他算法（pulldiag, frsd, frozen）】

# 图1：n=20, row_and_col_mat(n=20, p=0.5）
# 图2：n=20, row_and_col_mat(n=20, p=0.2）
# 图3:  n=20, di_ring(n=20)
# 图4：三角网格图 或者我提供的 grid_20()


# 三、【噪声取1e-4，强异质性，只看pullsum】
# 图1：pull sum 在row_and_col_mat(n=20, p=0.5), row_and_col_mat(n=20, p=0.2）,di_ring(n=20),grid_20() 这些拓扑下的表现画在一张图内





#下列所有代码会返回一个行随机矩阵和一个列随机矩阵。它们的对角元可以达到$2^{-n}$量级，$\kappa_\pi$为$n$的量级。

def row_and_col_mat(n=10, p=0.3, seed=None,show_graph=None): #生成p-随机图，一般比较好。
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 生成强连通的随机有向图
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    
    # 确保图是强连通的
    while not nx.is_strongly_connected(G):
        G = nx.gnp_random_graph(n, p, directed=True)
    
    # 计算每个节点的入度和出度
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # 初始化权重矩阵
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    # 给每条边赋予权重 1 / (目标节点的入度 + 1)
    for (i, j) in G.edges():
        A[i][j] = 1 / (in_degrees[j] + 1)
        B[i][j] = 1 / (out_degrees[i] + 1)
    # 为每个节点添加自环，并计算自环权重 1 / (入度 + 1)
    for j in range(n):
        A[j][j] = 1 / (in_degrees[j] + 1)
        B[j][j] = 1 / (out_degrees[j] + 1)
    if show_graph is not None:
        return A.T,B.T,G
    return A.T, B.T #A.T是行随机，B.T是列随机矩阵，

def di_ring(n=10): #生成稀疏环状图。也可以取n=5
    A,B=np.eye(n)/2,np.eye(n)/2
    m=int(n/2)
    for i in range(n-1):
        A[i][i+1]=0.5
        B[i][i+1]=0.5
    A[n-1][0]=0.5; B[n-1][0]=0.5
    A[0][m]=1/3
    A[m-1][m]=1/3
    A[m][m]=1/3
    B[0][0]=1/3
    B[0][1]=1/3
    B[0][m]=1/3
    return A.T,B.T #A.T是行随机，B.T是列随机矩阵，

def grid_10(): #生成10节点稀疏网格图。
    A=np.eye(10)/2
    A[0][1]=1/2;
    A[3][2]=1/3;A[2][2]=1/3;A[1][2]=1/3;
    A[2][7]=1/2;A[7][8]=1/2;A[8][9]=1/2;A[9][0]=1/2;
    A[7][6]=1/2;A[6][5]=1/2;A[4][3]=1/2;A[5][4]=1/2;
    B=np.eye(10)/2
    B[0][1]=1/2;B[1][2]=1/2;B[2][7]=1/2;B[3][2]=1/2
    B[4][3]=1/2;B[5][4]=1/2;B[6][5]=1/2;
    B[7][6]=1/3;B[7][7]=1/3;B[7][8]=1/3
    B[8][9]=1/2;B[9][0]=1/2
    return A.T,B.T #A.T是行随机，B.T是列随机矩阵，



def grid_20():  # 生成20节点稀疏网格图
    A = np.eye(20) / 2
    # 定义A矩阵的非对角线元素
    A[0][1] = 1/2; A[1][2] = 1/3; A[2][3] = 1/3; A[3][4] = 1/3
    A[4][5] = 1/2; A[5][6] = 1/2; A[6][7] = 1/2; A[7][8] = 1/2
    A[8][9] = 1/2; A[9][10] = 1/2; A[10][11] = 1/2; A[11][12] = 1/2
    A[12][13] = 1/2; A[13][14] = 1/2; A[14][15] = 1/2; A[15][16] = 1/2
    A[16][17] = 1/2; A[17][18] = 1/2; A[18][19] = 1/2; A[19][0] = 1/2

    # 再添加一些其他稀疏的连接
    A[2][10] = 1/2; A[10][15] = 1/2; A[15][5] = 1/2
    
    B = np.eye(20) / 2
    # 定义B矩阵的非对角线元素
    B[0][1] = 1/2; B[1][2] = 1/2; B[2][3] = 1/2; B[3][4] = 1/2
    B[4][5] = 1/2; B[5][6] = 1/2; B[6][7] = 1/2; B[7][8] = 1/3
    B[8][9] = 1/3; B[9][10] = 1/3; B[10][11] = 1/2; B[11][12] = 1/2
    B[12][13] = 1/2; B[13][14] = 1/2; B[14][15] = 1/2; B[15][16] = 1/2
    B[16][17] = 1/2; B[17][18] = 1/2; B[18][19] = 1/2; B[19][0] = 1/2
    
    # 再添加一些其他稀疏的连接
    B[5][10] = 1/3; B[10][15] = 1/3; B[15][5] = 1/3
    
    return Row(A.T), Col(B.T)  

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
    # 计算每一列的和
    col_sums = np.sum(W, axis=0)  
  
    # 将每一列除以该列的和  
    for i in range(W.shape[0]):  
        W[:, i] /= col_sums[i]  
  
    return W

def get_xinmeng_matrix(n=5):
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
