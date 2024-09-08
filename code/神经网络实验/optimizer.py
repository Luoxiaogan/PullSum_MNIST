import torch
import copy
from torch.optim.optimizer import Optimizer 

class PullSum(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(model_list[0].parameters().__next__().device)  # 确保A在同一设备上
        self.B = B.to(model_list[0].parameters().__next__().device)  # 确保B在同一设备上

        closure()  # 先计算一遍梯度
            
        self.prev_model_list = [copy.deepcopy(model) for model in model_list]
        for i, model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param, param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                if param.grad is not None:
                    prev_param.grad = param.grad.clone()

        self.v_list = []
        for model in self.model_list:
            model_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.prev_v_list = [[v.clone() if v is not None else None for v in model_gradients] for model_gradients in self.v_list]
        
        self.correction_vector = torch.ones(A.shape[0], device=model_list[0].parameters().__next__().device)  # 确保在相同设备上

        defaults = dict(lr=lr)
        super(PullSum, self).__init__(model_list[0].parameters(), defaults)
    
    def step(self, closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")

        with torch.no_grad():
            self.correction_vector = torch.matmul(self.A.T, self.correction_vector)  # 使用CUDA设备的矩阵运算
            # step1: x = Ax
            for i, model in enumerate(self.model_list):
                for params, prev_params in zip(model.parameters(), zip(*[m.parameters() for m in self.prev_model_list])):
                    weighted_sum = torch.zeros_like(params)
                    for j, prev_param in enumerate(prev_params):
                        weighted_sum += self.A[i, j] * prev_param
                    params.data.copy_(weighted_sum)
            
            # step2: x = x - lr * (1 / correction) * v
            for i, model in enumerate(self.model_list):
                for param, v in zip(model.parameters(), self.v_list[i]):
                    if v is not None:
                        correction = self.correction_vector[i]
                        update = self.lr * (1 / correction) * v
                        param.data -= update

        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播
        
        with torch.no_grad():
            # step4: v = Bv + g - prev_g
            new_v_list = []
            for i, model in enumerate(self.model_list):
                new_v = []
                for idx, (param, prev_param) in enumerate(zip(model.parameters(), self.prev_model_list[i].parameters())):
                    if param.grad is not None:
                        weighted_v_sum = torch.zeros_like(param.grad)
                        for j in range(len(self.model_list)):
                            weighted_v_sum += self.B[i, j] * self.v_list[j][idx]
                        v_update = weighted_v_sum + param.grad - prev_param.grad
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)
            
            self.v_list = new_v_list
            
            # step5
            for i, model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param, param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                    if param.grad is not None:
                        prev_param.grad = param.grad.clone()
            
            self.prev_v_list = [[v.clone() if v is not None else None for v in self.v_list[i]] for i in range(len(self.model_list))]

        return loss

class PullDiag(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(model_list[0].parameters().__next__().device)  # 确保A在同一设备上

        closure()  # 先计算一遍梯度
            
        self.prev_model_list = [copy.deepcopy(model) for model in model_list]
        for i, model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param, param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                if param.grad is not None:
                    prev_param.grad = param.grad.clone()

        self.v_list = []
        for model in self.model_list:
            model_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.prev_v_list = [[v.clone() if v is not None else None for v in model_gradients] for model_gradients in self.v_list]

        device = model_list[0].parameters().__next__().device
        self.W = torch.eye(A.shape[0], device=device)  # 确保W在同一设备上
        self.prev_W = torch.eye(A.shape[0], device=device)  # 确保prev_W在同一设备上

        defaults = dict(lr=lr)
        super(PullDiag, self).__init__(model_list[0].parameters(), defaults)
    
    def step(self, closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")
        
        with torch.no_grad():
            self.W = self.A @ self.W  # 确保矩阵乘法在相同设备上
            # step1: x = Ax
            for i, model in enumerate(self.model_list):
                for params, prev_params in zip(model.parameters(), zip(*[m.parameters() for m in self.prev_model_list])):
                    weighted_sum = torch.zeros_like(params)
                    for j, prev_param in enumerate(prev_params):
                        weighted_sum += self.A[i, j] * prev_param
                    params.data.copy_(weighted_sum)
            
            # step2: x = x - lr * v
            for i, model in enumerate(self.model_list):
                for param, v in zip(model.parameters(), self.v_list[i]):
                    if v is not None:
                        update = self.lr * v
                        param.data -= update
        
        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播
        
        with torch.no_grad():
            # step4: v = Av + W @ g - prev_W @ prev_g
            new_v_list = []
            for i, model in enumerate(self.model_list):
                new_v = []
                for idx, (param, prev_param) in enumerate(zip(model.parameters(), self.prev_model_list[i].parameters())):
                    if param.grad is not None and prev_param.grad is not None:
                        weighted_v_sum = torch.zeros_like(param.grad)
                        for j in range(len(self.model_list)):
                            weighted_v_sum += self.A[i, j] * self.v_list[j][idx]
                        v_update = (weighted_v_sum + (1 / self.W[i, i]) * param.grad - (1 / self.prev_W[i, i]) * prev_param.grad)
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)
            
            self.v_list = new_v_list
            # step5
            self.prev_W = self.W.clone()  # 确保prev_W在同一设备上
            for i, model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param, param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                    if param.grad is not None:
                        prev_param.grad = param.grad.clone()
            self.prev_v_list = [[v.clone() if v is not None else None for v in self.v_list[i]] for i in range(len(self.model_list))]

        return loss
    
class FRSD(Optimizer):
    def __init__(self,model_list,lr=1e-2,beta=0.1,d=784,A=None,closure=None):
        self.model_list=model_list
        self.lr=lr
        self.beta=beta
        self.A = A.to(model_list[0].parameters().__next__().device)

        closure()

        self.v_list = []
        for model in self.model_list:
            model_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.u_list = []
        for model in self.model_list:
            model_u=[]
            for param in model.parameters():
                model_u.append(torch.zeros_like(param, device=param.device))
            self.u_list.append(model_u)
        
        self.z_list = []
        for model in self.model_list:
            model_z=[]
            for param in model.parameters():
                model_z.append(torch.zeros_like(param, device=param.device))
            self.z_list.append(model_z)
        
        self.W = torch.eye(A.shape[0], device=model_list[0].parameters().__next__().device)

        defaults = dict(lr=lr)
        super(FRSD, self).__init__(model_list[0].parameters(), defaults)

    def step(self,closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
            
        with torch.no_grad():
            # step1: z = Ax
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 初始化 z_list[i][param_idx] 为零张量
                    self.z_list[i][param_idx].zero_()
                    
                    # 获取所有模型的对应参数
                    for j, other_model in enumerate(self.model_list):
                        self.z_list[i][param_idx].add_(self.A[i, j] * list(other_model.parameters())[param_idx])
                    
                    # 确保 z_list[i][param_idx] 的 device 与 param 一致
                    self.z_list[i][param_idx] = self.z_list[i][param_idx].to(param.device)
            #step2: u=u+beta*(x-z)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 更新 u_list[i][param_idx] 参数
                    self.u_list[i][param_idx].add_(self.beta * (param - self.z_list[i][param_idx]))
        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播   

        with torch.no_grad():
            #step4: v=np.linalg.inv(np.diag(np.diag(W)))@g
            for i, model in enumerate(self.model_list):
                diag_inv = 1.0 / self.W[i, i]  # W 的第 i 个对角元的倒数
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        # 更新 v_list[i][param_idx]
                        self.v_list[i][param_idx].copy_(diag_inv * param.grad)
            #step5: x=z-lr*(v+u)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 获取对应的 z, v, u 参数
                    z = self.z_list[i][param_idx]
                    v = self.v_list[i][param_idx]
                    u = self.u_list[i][param_idx]
                    
                    # 更新模型参数 x: x = z - lr * (v + u)
                    param.copy_(z - self.lr * (v + u))
            #step6: W=W@A
            self.W = self.W @ self.A

        return loss

class FROZEN(Optimizer):
    def __init__(self,model_list,lr=1e-2,beta=0.1,d=784,A=None,closure=None):
        self.model_list=model_list
        self.lr=lr
        self.beta=beta
        self.A = A.to(model_list[0].parameters().__next__().device)

        closure()

        self.v_list = []
        for model in self.model_list:
            model_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.s_list = []
        self.prev_s_list = []
        for model in self.model_list:
            model_s = []
            for param in model.parameters():
                model_s.append(torch.zeros_like(param, device=param.device))
            self.s_list.append(model_s)
            self.prev_s_list.append([torch.zeros_like(param, device=param.device) for param in model.parameters()])

        self.prev_g_list = []
        for model in self.model_list:
            model_prev_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_prev_gradients.append(param.grad.clone())
                else:
                    model_prev_gradients.append(torch.zeros_like(param, device=param.device))
            self.prev_g_list.append(model_prev_gradients)    
    
        self.W = torch.eye(A.shape[0], device=model_list[0].parameters().__next__().device)

        defaults = dict(lr=lr)
        super(FROZEN, self).__init__(model_list[0].parameters(), defaults)

    def step(self,closure):
        with torch.no_grad():
            #step1: pre_s=s, s=A@x-lr*v
            for i, model in enumerate(self.model_list):
                s_update = []  # 用于存储第 i 个模型的 s_list 更新值

                for param_idx, param in enumerate(model.parameters()):
                    # 计算 A 的第 i 行的加权平均
                    A_row = self.A[i]
                    weighted_sum = sum(A_row[j] * list(self.model_list[j].parameters())[param_idx] for j in range(len(self.model_list)))
                    
                    # 更新 s = A @ x - lr * v
                    updated_s = weighted_sum - self.lr*self.v_list[i][param_idx]
                    
                    s_update.append(updated_s)
                
                # 更新 s_list 中的对应元素
                self.prev_s_list[i] = self.s_list[i]  # 保存上一步的 s_list
                self.s_list[i] = s_update
            #step2: x=s+beta*(s-s_pre)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 使用 s_list 和 prev_s_list 更新模型参数 x
                    updated_x = self.s_list[i][param_idx] + self.beta * (self.s_list[i][param_idx] - self.prev_s_list[i][param_idx])
                    
                    # 更新模型的参数
                    param.copy_(updated_x)
        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播   
        with torch.no_grad():
            #step4: v=A@v, 并且能保证之前的更改不会影响后面的更改
            temp_v_list = [list(v) for v in self.v_list]  # 临时存储 v_list 的副本
            for i, model in enumerate(self.model_list):
                diag_inv = 1.0 / self.W[i, i]  # W 的第 i 个对角元的倒数
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        # 计算 v_update
                        g = param.grad
                        prev_g = self.prev_g_list[i][param_idx] if self.prev_g_list[i][param_idx] is not None else torch.zeros_like(g)
                        v_update = diag_inv * g + (1.0 - diag_inv) * prev_g
                        temp_v_list[i][param_idx] = v_update
                    else:
                        temp_v_list[i][param_idx] = torch.zeros_like(param.grad) if param.grad is not None else None

            # 更新 v_list
            self.v_list = temp_v_list
            #step5 : v+=?@g-?@prev_g 同时更新prev_g为当前梯度
            for i,model in enumerate(self.model_list):
                diag_inv_old = 1.0/self.W[i,i]
                #同时更新W
                self.W=self.W@self.A
                diag_inv_new = 1.0/self.W[i,i]
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        g=param.grad 
                        prev_g = self.prev_g_list[i][param_idx] if self.prev_g_list[i][param_idx] is not None else torch.zeros_like(g)
                        v_update = diag_inv_new*g-diag_inv_old*prev_g
                        self.v_list[i][param_idx] += v_update
                    else:
                        self.v_list[i][param_idx] = torch.zeros_like(param.grad) if param.grad is not None else None
                self.prev_g_list[i] = [param.grad.clone() if param.grad is not None else None for param in model.parameters()]
        return loss

class Diag_SGD(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(model_list[0].parameters().__next__().device)  # 确保A在同一设备上

        closure()  # 先计算一遍梯度

        self.v_list = []
        for model in self.model_list:
            model_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        device = model_list[0].parameters().__next__().device
        self.W = torch.eye(A.shape[0], device=device)  # 确保W在同一设备上

        defaults = dict(lr=lr)
        super(Diag_SGD, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure):
        for model in self.model_list:
            model.zero_grad()
            loss = closure()
        with torch.no_grad():
            #一步完成更新v和x=Ax-lr*v
            for i, model in enumerate(self.model_list):
                # Update v_list with scaled gradients
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        self.v_list[i][j] = (1.0 / self.W[i, i]) * param.grad

                # Update each parameter
                for j, param in enumerate(model.parameters()):
                    avg_params = torch.zeros_like(param)
                    for k, other_model in enumerate(self.model_list):
                        avg_params += self.A[i, k] * list(other_model.parameters())[j]
                    
                    if self.v_list[i][j] is not None:
                        param.data = avg_params - self.lr * self.v_list[i][j]

            # 更新W
            self.W = self.W @ self.A

        return loss