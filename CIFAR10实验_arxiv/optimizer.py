import torch
import copy
from torch.optim.optimizer import Optimizer


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


class PullDiag(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store previous parameters and gradients as vectors
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

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        # Initialize w_vector and prev_w_vector
        self.w_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )
        self.prev_w_vector = self.w_vector.clone()

        defaults = dict(lr=lr)
        super(PullDiag, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        with torch.no_grad():
            # Update w_vector
            self.w_vector = torch.matmul(self.A, self.w_vector)

            # Step1: x = Ax
            prev_params_tensor = torch.stack(self.prev_params)  # (n_models, param_size)
            new_params_tensor = torch.matmul(
                self.A, prev_params_tensor
            )  # (n_models, param_size)

            # Step2: x = x - lr * v
            v_tensor = torch.stack(self.v_list)  # (n_models, param_size)
            new_params_tensor -= self.lr * v_tensor

            # Update model parameters
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # Step3: compute new gradients
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Compute new gradients
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            # Step4: v = A v + (1 / w_vector) * g - (1 / prev_w_vector) * prev_g
            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            weighted_v = torch.matmul(self.A, v_tensor)
            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)  # Shape (n_models, 1)
            prev_w_vector_inv = 1.0 / self.prev_w_vector.unsqueeze(1)

            W_g = w_vector_inv * new_grads_tensor
            prev_W_prev_g = prev_w_vector_inv * prev_grads_tensor
            new_v_tensor = weighted_v + W_g - prev_W_prev_g

            # Update v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # Step5: Update prev_params, prev_grads, prev_w_vector
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_w_vector = self.w_vector.clone()

        return loss


class FRSD(Optimizer):
    def __init__(self, model_list, lr=1e-2, beta=0.1, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.beta = beta
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store parameters and gradients as vectors
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

        # Initialize v_list, u_list, z_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]
        param_size = self.prev_params[0].numel()
        device = self.prev_params[0].device
        self.u_list = [torch.zeros(param_size, device=device) for _ in self.model_list]
        self.z_list = [torch.zeros(param_size, device=device) for _ in self.model_list]

        # Initialize w_vector
        self.w_vector = torch.ones(self.A.shape[0], device=device)

        defaults = dict(lr=lr)
        super(FRSD, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        with torch.no_grad():
            # Update w_vector
            self.w_vector = torch.matmul(self.w_vector, self.A)

            # Step1: z = A x
            prev_params_tensor = torch.stack(self.prev_params)  # (n_models, param_size)
            z_tensor = torch.matmul(self.A, prev_params_tensor)

            # Step2: u = u + beta * (prev_x - z)
            prev_params_tensor = torch.stack(self.prev_params)
            u_tensor = torch.stack(self.u_list)
            u_tensor += self.beta * (prev_params_tensor - z_tensor)

        # Step3: compute gradients
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Compute new gradients
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            # Step4: v = diag_inv(W) @ g
            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)  # Shape (n_models, 1)
            v_tensor = w_vector_inv * new_grads_tensor

            # Step5: x = z - lr * (v + u)
            x_tensor = z_tensor - self.lr * (v_tensor + u_tensor)

            # Update model parameters
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(x_tensor[i], model.parameters())

            # Update variables
            self.prev_params = [
                x_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.u_list = [u_tensor[i].clone() for i in range(len(self.model_list))]
            self.v_list = [v_tensor[i].clone() for i in range(len(self.model_list))]

        return loss


class FROZEN(Optimizer):
    def __init__(self, model_list, lr=1e-2, beta=0.1, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.beta = beta
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store parameters and gradients as vectors
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

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        # Initialize s_list and prev_s_list
        param_size = self.prev_params[0].numel()
        device = self.prev_params[0].device
        self.s_list = [torch.zeros(param_size, device=device) for _ in self.model_list]
        self.prev_s_list = [
            torch.zeros(param_size, device=device) for _ in self.model_list
        ]

        # Initialize w_vector
        self.w_vector = torch.ones(self.A.shape[0], device=device)

        defaults = dict(lr=lr)
        super(FROZEN, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        with torch.no_grad():
            # Step6: Update w_vector
            self.w_vector = torch.matmul(self.w_vector, self.A)

            # Step1: prev_s = s, s = A x - lr * v
            prev_params_tensor = torch.stack(self.prev_params)
            prev_s_tensor = torch.stack(self.s_list)
            self.prev_s_list = [
                prev_s_tensor[i].clone() for i in range(len(self.model_list))
            ]
            s_tensor = torch.matmul(self.A, prev_params_tensor) - self.lr * torch.stack(
                self.v_list
            )
            self.s_list = [s_tensor[i].clone() for i in range(len(self.model_list))]

            # Step2: x = s + beta * (s - prev_s)
            x_tensor = s_tensor + self.beta * (s_tensor - prev_s_tensor)

            # Update model parameters
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(x_tensor[i], model.parameters())

        # Step3: compute gradients
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Compute new gradients
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            # Step4: v = A v
            v_tensor = torch.matmul(self.A, torch.stack(self.v_list))

            # Update v: v += diag_inv_new * g - diag_inv_old * prev_g
            prev_grads_tensor = torch.stack(self.prev_grads)
            w_vector_inv_old = 1.0 / self.w_vector.unsqueeze(
                1
            )  # Before updating w_vector
            self.w_vector = torch.matmul(self.w_vector, self.A)  # Update w_vector
            w_vector_inv_new = 1.0 / self.w_vector.unsqueeze(
                1
            )  # After updating w_vector
            v_tensor += (
                w_vector_inv_new * new_grads_tensor
                - w_vector_inv_old * prev_grads_tensor
            )

            # Update variables
            self.v_list = [v_tensor[i].clone() for i in range(len(self.model_list))]
            self.prev_params = [
                x_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]

        return loss
