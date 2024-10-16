import torch
import copy
from torch.optim.optimizer import Optimizer


class PullSum_for_try(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to("cuda:0")
        self.B = A.to("cuda:0")

        closure()  # 先计算一遍梯度

        self.prev_model_list = [copy.deepcopy(model) for model in model_list]
        for i, model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param, param in zip(
                self.prev_model_list[i].parameters(), model.parameters()
            ):
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

        self.prev_v_list = [
            [v.clone() if v is not None else None for v in model_gradients]
            for model_gradients in self.v_list
        ]

        self.correction_vector = torch.ones(A.shape[0], device=A.device)

        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")

        defaults = dict(lr=lr)
        super(PullSum_for_try, self).__init__(model_list[0].parameters(), defaults)

        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")

    def step(self, closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")

        with torch.no_grad():
            self.correction_vector = torch.matmul(
                self.A.T, self.correction_vector
            )  # 使用CUDA设备的矩阵运算
            # step1: x = Ax
            for i, model in enumerate(self.model_list):
                for params, prev_params in zip(
                    model.parameters(),
                    zip(*[m.parameters() for m in self.prev_model_list]),
                ):
                    weighted_sum = torch.zeros_like(params).to("cuda:0")
                    for j, prev_param in enumerate(prev_params):
                        prev_param_cuda0 = prev_param.to("cuda:0")
                        weighted_sum += self.A[i, j] * prev_param_cuda0
                    params.data.copy_(weighted_sum.to(params.device))
                    del prev_param_cuda0

            # step2: x = x - lr * (1 / correction) * v
            for i, model in enumerate(self.model_list):
                for param, v in zip(model.parameters(), self.v_list[i]):
                    if v is not None:
                        correction = self.correction_vector[i]
                        v_cuda0 = v.to("cuda:0")
                        update = self.lr * (1 / correction) * v_cuda0
                        param.data -= update.to(param.device)

        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播

        with torch.no_grad():
            # step4: v = Bv + g - prev_g
            new_v_list = []
            # 首先将 self.v_list 的每个子列表都转换到 cuda:0 上
            v_list_cuda0 = [
                [v_param.to("cuda:0") for v_param in self.v_list[j]]
                for j in range(len(self.model_list))
            ]
            for i, model in enumerate(self.model_list):
                new_v = []
                for idx, (param, prev_param) in enumerate(
                    zip(model.parameters(), self.prev_model_list[i].parameters())
                ):
                    if param.grad is not None:

                        weighted_v_sum = torch.zeros_like(param.grad).to("cuda:0")

                        for j in range(len(self.model_list)):
                            weighted_v_sum += self.B[i, j] * v_list_cuda0[j][idx]
                        weighted_v_sum = weighted_v_sum.to(param.device)
                        v_update = weighted_v_sum + param.grad - prev_param.grad
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)

            self.v_list = new_v_list

            # step5
            for i, model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param, param in zip(
                    self.prev_model_list[i].parameters(), model.parameters()
                ):
                    if param.grad is not None:
                        prev_param.grad = param.grad.clone()

            self.prev_v_list = [
                [v.clone() if v is not None else None for v in self.v_list[i]]
                for i in range(len(self.model_list))
            ]

        return loss


""" # 借助GPT改进的的PullSum优化器 ——第三个版本 
# 适用于混合精度计算    

class PullSum(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)
        self.B = B.to(next(model_list[0].parameters()).device)

        # 计算初始梯度
        closure()  # closure 应该是 closure_init

        # 将模型参数展平成向量
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector(
                [p.grad for p in model.parameters()]
            ).detach().clone()
            for model in self.model_list
        ]

        # 初始化 v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        self.correction_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )

        defaults = dict(lr=lr)
        super(PullSum, self).__init__(model_list[0].parameters(), defaults)

    def step(self):
        with torch.no_grad():
            # 更新修正向量
            self.correction_vector = torch.matmul(self.A.T, self.correction_vector)

            # 将 prev_params 和 v_list 堆叠成张量
            prev_params_tensor = torch.stack(self.prev_params)
            v_tensor = torch.stack(self.v_list)

            # Step 1: x = Ax
            new_params_tensor = torch.matmul(self.A, prev_params_tensor)

            # Step 2: x = x - lr * (1 / correction) * v
            correction_vector = self.correction_vector.unsqueeze(1)
            scaled_v = self.lr * v_tensor / correction_vector
            new_params_tensor -= scaled_v

            # 更新模型参数
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(new_params_tensor[i], model.parameters())

            # 提取当前梯度
            new_grads = []
            for model in self.model_list:
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().clone())
                    else:
                        grads.append(torch.zeros_like(p))
                new_grads.append(torch.nn.utils.parameters_to_vector(grads))
            new_grads_tensor = torch.stack(new_grads)

            # Step 4: v = Bv + g - prev_g
            weighted_v_tensor = torch.matmul(self.B, v_tensor)
            grad_diff_tensor = new_grads_tensor - torch.stack(self.prev_grads)
            new_v_tensor = weighted_v_tensor + grad_diff_tensor

            # 更新 v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # Step 5: 更新 prev_params 和 prev_grads
            self.prev_params = [new_params_tensor[i].clone() for i in range(len(self.model_list))]
            self.prev_grads = [new_grads_tensor[i].clone() for i in range(len(self.model_list))] """


# 借助GPT改进的的PullSum优化器 ——第二个版本


class PullSum(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)
        self.B = B.to(next(model_list[0].parameters()).device)

        # 计算初始梯度
        closure()

        # 将模型参数展平成向量
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        # 初始化 v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        self.correction_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )

        defaults = dict(lr=lr)
        super(PullSum, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # 可以在每一步step修改lr
        with torch.no_grad():
            # 更新修正向量
            self.correction_vector = torch.matmul(self.A.T, self.correction_vector)

            # 将 prev_params 和 v_list 堆叠成张量
            prev_params_tensor = torch.stack(
                self.prev_params
            )  # 形状：(n_models, param_size)
            v_tensor = torch.stack(self.v_list)  # 形状：(n_models, param_size)

            # Step 1: x = Ax
            new_params_tensor = torch.matmul(
                self.A, prev_params_tensor
            )  # 形状：(n_models, param_size)

            # Step 2: x = x - lr * (1 / correction) * v
            correction_vector = self.correction_vector.unsqueeze(
                1
            )  # 形状：(n_models, 1)
            scaled_v = (
                self.lr * v_tensor / correction_vector
            )  # 形状：(n_models, param_size)
            new_params_tensor -= scaled_v

            # 更新模型参数
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # Step 3: 计算新的梯度
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # 展平新的梯度
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)  # 形状：(n_models, param_size)

            # Step 4: v = Bv + g - prev_g
            weighted_v_tensor = torch.matmul(
                self.B, v_tensor
            )  # 形状：(n_models, param_size)
            grad_diff_tensor = new_grads_tensor - torch.stack(self.prev_grads)
            new_v_tensor = weighted_v_tensor + grad_diff_tensor

            # 更新 v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # Step 5: 更新 prev_params 和 prev_grads
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]

        return loss


# 借助GPT改进的的PullSum优化器 ——第一个版本
""" 
class PullSum(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)
        self.B = B.to(next(model_list[0].parameters()).device)

        # 计算初始梯度
        closure()

        # 存储上一轮的参数和梯度，避免深度拷贝整个模型
        self.prev_params = [
            [param.data.clone() for param in model.parameters()]
            for model in model_list
        ]
        self.prev_grads = [
            [param.grad.clone() if param.grad is not None else None for param in model.parameters()]
            for model in model_list
        ]

        # 初始化 v_list
        self.v_list = [
            [param.grad.clone() if param.grad is not None else None for param in model.parameters()]
            for model in model_list
        ]

        self.correction_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )

        defaults = dict(lr=lr)
        super(PullSum, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr#可以在每一步step修改lr
        with torch.no_grad():
            # 更新修正向量
            self.correction_vector = torch.matmul(self.A.T, self.correction_vector)

            # Step 1: x = Ax
            for i, model in enumerate(self.model_list):
                param_iter = model.parameters()
                for idx, param in enumerate(param_iter):
                    weighted_sum = self.A[i, 0] * self.prev_params[0][idx]
                    for j in range(1, len(self.model_list)):
                        weighted_sum += self.A[i, j] * self.prev_params[j][idx]
                    param.data.copy_(weighted_sum)

            # Step 2: x = x - lr * (1 / correction) * v
            for i, model in enumerate(self.model_list):
                correction = self.correction_vector[i]
                for idx, param in enumerate(model.parameters()):
                    v = self.v_list[i][idx]
                    if v is not None:
                        param.data -= self.lr * (v / correction)

        # Step 3: 计算新的梯度
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Step 4: v = Bv + g - prev_g
            new_v_list = []
            for i, model in enumerate(self.model_list):
                new_v = []
                for idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        weighted_v = self.B[i, 0] * self.v_list[0][idx]
                        for j in range(1, len(self.model_list)):
                            weighted_v += self.B[i, j] * self.v_list[j][idx]
                        v_update = weighted_v + param.grad - self.prev_grads[i][idx]
                        new_v.append(v_update)
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)
            self.v_list = new_v_list

            # Step 5: 更新 prev_params 和 prev_grads
            self.prev_params = [
                [param.data.clone() for param in model.parameters()]
                for model in self.model_list
            ]
            self.prev_grads = [
                [param.grad.clone() if param.grad is not None else None for param in model.parameters()]
                for model in self.model_list
            ]

        return loss """


""" class PullSum(Optimizer):
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

        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")

        defaults = dict(lr=lr)
        super(PullSum, self).__init__(model_list[0].parameters(), defaults)
        
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")
    
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

        return loss """


class PullDiag(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(model_list[0].parameters().__next__().device)  # 确保A在同一设备上

        closure()  # 先计算一遍梯度

        self.prev_model_list = [copy.deepcopy(model) for model in model_list]
        for i, model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param, param in zip(
                self.prev_model_list[i].parameters(), model.parameters()
            ):
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

        self.prev_v_list = [
            [v.clone() if v is not None else None for v in model_gradients]
            for model_gradients in self.v_list
        ]

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
                for params, prev_params in zip(
                    model.parameters(),
                    zip(*[m.parameters() for m in self.prev_model_list]),
                ):
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
                for idx, (param, prev_param) in enumerate(
                    zip(model.parameters(), self.prev_model_list[i].parameters())
                ):
                    if param.grad is not None and prev_param.grad is not None:
                        weighted_v_sum = torch.zeros_like(param.grad)
                        for j in range(len(self.model_list)):
                            weighted_v_sum += self.A[i, j] * self.v_list[j][idx]
                        v_update = (
                            weighted_v_sum
                            + (1 / self.W[i, i]) * param.grad
                            - (1 / self.prev_W[i, i]) * prev_param.grad
                        )
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)

            self.v_list = new_v_list
            # step5
            self.prev_W = self.W.clone()  # 确保prev_W在同一设备上
            for i, model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param, param in zip(
                    self.prev_model_list[i].parameters(), model.parameters()
                ):
                    if param.grad is not None:
                        prev_param.grad = param.grad.clone()
            self.prev_v_list = [
                [v.clone() if v is not None else None for v in self.v_list[i]]
                for i in range(len(self.model_list))
            ]

        return loss


class FRSD(Optimizer):
    def __init__(self, model_list, lr=1e-2, beta=0.1, d=784, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.beta = beta
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
            model_u = []
            for param in model.parameters():
                model_u.append(torch.zeros_like(param, device=param.device))
            self.u_list.append(model_u)

        self.z_list = []
        for model in self.model_list:
            model_z = []
            for param in model.parameters():
                model_z.append(torch.zeros_like(param, device=param.device))
            self.z_list.append(model_z)

        self.W = torch.eye(
            A.shape[0], device=model_list[0].parameters().__next__().device
        )

        defaults = dict(lr=lr)
        super(FRSD, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure):
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
                        self.z_list[i][param_idx].add_(
                            self.A[i, j] * list(other_model.parameters())[param_idx]
                        )

                    # 确保 z_list[i][param_idx] 的 device 与 param 一致
                    self.z_list[i][param_idx] = self.z_list[i][param_idx].to(
                        param.device
                    )
            # step2: u=u+beta*(x-z)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 更新 u_list[i][param_idx] 参数
                    self.u_list[i][param_idx].add_(
                        self.beta * (param - self.z_list[i][param_idx])
                    )
        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播

        with torch.no_grad():
            # step4: v=np.linalg.inv(np.diag(np.diag(W)))@g
            for i, model in enumerate(self.model_list):
                diag_inv = 1.0 / self.W[i, i]  # W 的第 i 个对角元的倒数
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        # 更新 v_list[i][param_idx]
                        self.v_list[i][param_idx].copy_(diag_inv * param.grad)
            # step5: x=z-lr*(v+u)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 获取对应的 z, v, u 参数
                    z = self.z_list[i][param_idx]
                    v = self.v_list[i][param_idx]
                    u = self.u_list[i][param_idx]

                    # 更新模型参数 x: x = z - lr * (v + u)
                    param.copy_(z - self.lr * (v + u))
            # step6: W=W@A
            self.W = self.W @ self.A

        return loss


class FROZEN(Optimizer):
    def __init__(self, model_list, lr=1e-2, beta=0.1, d=784, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.beta = beta
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
            self.prev_s_list.append(
                [
                    torch.zeros_like(param, device=param.device)
                    for param in model.parameters()
                ]
            )

        self.prev_g_list = []
        for model in self.model_list:
            model_prev_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    model_prev_gradients.append(param.grad.clone())
                else:
                    model_prev_gradients.append(
                        torch.zeros_like(param, device=param.device)
                    )
            self.prev_g_list.append(model_prev_gradients)

        self.W = torch.eye(
            A.shape[0], device=model_list[0].parameters().__next__().device
        )

        defaults = dict(lr=lr)
        super(FROZEN, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure):
        with torch.no_grad():
            # step1: pre_s=s, s=A@x-lr*v
            for i, model in enumerate(self.model_list):
                s_update = []  # 用于存储第 i 个模型的 s_list 更新值

                for param_idx, param in enumerate(model.parameters()):
                    # 计算 A 的第 i 行的加权平均
                    A_row = self.A[i]
                    weighted_sum = sum(
                        A_row[j] * list(self.model_list[j].parameters())[param_idx]
                        for j in range(len(self.model_list))
                    )

                    # 更新 s = A @ x - lr * v
                    updated_s = weighted_sum - self.lr * self.v_list[i][param_idx]

                    s_update.append(updated_s)

                # 更新 s_list 中的对应元素
                self.prev_s_list[i] = self.s_list[i]  # 保存上一步的 s_list
                self.s_list[i] = s_update
            # step2: x=s+beta*(s-s_pre)
            for i, model in enumerate(self.model_list):
                for param_idx, param in enumerate(model.parameters()):
                    # 使用 s_list 和 prev_s_list 更新模型参数 x
                    updated_x = self.s_list[i][param_idx] + self.beta * (
                        self.s_list[i][param_idx] - self.prev_s_list[i][param_idx]
                    )

                    # 更新模型的参数
                    param.copy_(updated_x)
        # step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss = closure()  # 自动完成了反向传播
        with torch.no_grad():
            # step4: v=A@v, 并且能保证之前的更改不会影响后面的更改
            temp_v_list = [list(v) for v in self.v_list]  # 临时存储 v_list 的副本
            for i, model in enumerate(self.model_list):
                diag_inv = 1.0 / self.W[i, i]  # W 的第 i 个对角元的倒数
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        # 计算 v_update
                        g = param.grad
                        prev_g = (
                            self.prev_g_list[i][param_idx]
                            if self.prev_g_list[i][param_idx] is not None
                            else torch.zeros_like(g)
                        )
                        v_update = diag_inv * g + (1.0 - diag_inv) * prev_g
                        temp_v_list[i][param_idx] = v_update
                    else:
                        temp_v_list[i][param_idx] = (
                            torch.zeros_like(param.grad)
                            if param.grad is not None
                            else None
                        )

            # 更新 v_list
            self.v_list = temp_v_list
            # step5 : v+=?@g-?@prev_g 同时更新prev_g为当前梯度
            for i, model in enumerate(self.model_list):
                diag_inv_old = 1.0 / self.W[i, i]
                # 同时更新W
                self.W = self.W @ self.A
                diag_inv_new = 1.0 / self.W[i, i]
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        g = param.grad
                        prev_g = (
                            self.prev_g_list[i][param_idx]
                            if self.prev_g_list[i][param_idx] is not None
                            else torch.zeros_like(g)
                        )
                        v_update = diag_inv_new * g - diag_inv_old * prev_g
                        self.v_list[i][param_idx] += v_update
                    else:
                        self.v_list[i][param_idx] = (
                            torch.zeros_like(param.grad)
                            if param.grad is not None
                            else None
                        )
                self.prev_g_list[i] = [
                    param.grad.clone() if param.grad is not None else None
                    for param in model.parameters()
                ]
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
            # 一步完成更新v和x=Ax-lr*v
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
