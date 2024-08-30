import torch
import copy
from torch.optim.optimizer import Optimizer 

class PullSum(Optimizer):
    def __init__(self,model_list,lr=1e-2,A=None,B=None,closure=None):
        self.model_list=model_list
        self.lr=lr
        self.A=A
        self.B=B

        closure()#先计算一遍梯度
            
        self.prev_model_list=[copy.deepcopy(model) for model in model_list]
        for i,model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param,param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                if param.grad is not None:
                    prev_param.grad=param.grad.clone()

        self.v_list=[]
        for model in self.model_list:
            model_gradients=[]
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.prev_v_list = [[v.clone() if v is not None else None for v in model_gradients] for model_gradients in self.v_list]
        
        self.correction_vector=torch.ones(A.shape[0])

        defaults=dict(lr=lr)
        super(PullSum,self).__init__(model_list[0].parameters(),defaults)
    
    def step(self,closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")
        #loss=closure()
        with torch.no_grad():
            self.correction_vector=torch.matmul(self.A.T, self.correction_vector)
            # step1: x=Ax
            for i,model in enumerate(self.model_list):
                for params,prev_params in zip(model.parameters(),zip(*[m.parameters() for m in self.prev_model_list])):
                    weighted_sum=torch.zeros_like(params)
                    for j,prev_param in enumerate(prev_params):
                        weighted_sum+=self.A[i,j]*prev_param
                    params.data.copy_(weighted_sum)
            #step2: x=x-lr*(1/correction)*v
            for i,model in enumerate(self.model_list):
                for param,v in zip(model.parameters(),self.v_list[i]):
                    if v is not None:
                        correction=self.correction_vector[i]
                        update=self.lr*(1/correction)*v
                        param.data-=update
        #step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss=closure()#自动完成了反向传播
        with torch.no_grad():
            #step4: v=Bv+g-prev_g
            new_v_list=[]
            for i,model in enumerate(self.model_list):
                new_v=[]
                for idx,(param,prev_param) in enumerate(zip(model.parameters(),self.prev_model_list[i].parameters())):
                    if param.grad is not None:
                        weighted_v_sum=torch.zeros_like(param.grad)
                        for j in range(len(self.model_list)):
                            weighted_v_sum+=self.B[i,j]*self.v_list[j][idx]
                        v_update=weighted_v_sum+param.grad-prev_param.grad
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)
            self.v_list=new_v_list
            #step5
            for i,model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param,param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                    if param.grad is not None:
                        prev_param.grad=param.grad.clone()
            self.prev_v_list = [[v.clone() if v is not None else None for v in self.v_list[i]] for i in range(len(self.model_list))]

        return loss
    
class PullDiag(Optimizer):
    def __init__(self,model_list,lr=1e-2,A=None,closure=None):
        self.model_list=model_list
        self.lr=lr
        self.A=A

        closure()#先计算一遍梯度
            
        self.prev_model_list=[copy.deepcopy(model) for model in model_list]
        for i,model in enumerate(self.model_list):
            self.prev_model_list[i].load_state_dict(model.state_dict())
            for prev_param,param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                if param.grad is not None:
                    prev_param.grad=param.grad.clone()

        self.v_list=[]
        for model in self.model_list:
            model_gradients=[]
            for param in model.parameters():
                if param.grad is not None:
                    model_gradients.append(param.grad.clone())
                else:
                    model_gradients.append(None)
            self.v_list.append(model_gradients)

        self.prev_v_list = [[v.clone() if v is not None else None for v in model_gradients] for model_gradients in self.v_list]

        self.W=torch.eye(A.shape[0])
        self.prev_W=torch.eye(A.shape[0])

        defaults=dict(lr=lr)
        super(PullDiag,self).__init__(model_list[0].parameters(),defaults)
    
    def step(self,closure):
        for model_gradients in self.v_list:
            if any(v is None for v in model_gradients):
                raise ValueError("v_list contains None")
        for prev_model_gradients in self.prev_v_list:
            if any(v is None for v in prev_model_gradients):
                raise ValueError("prev_v_list contains None")
        #loss=closure()
        with torch.no_grad():
            self.W=self.A@self.W
            # step1: x=Ax
            for i,model in enumerate(self.model_list):
                for params,prev_params in zip(model.parameters(),zip(*[m.parameters() for m in self.prev_model_list])):
                    weighted_sum=torch.zeros_like(params)
                    for j,prev_param in enumerate(prev_params):
                        weighted_sum+=self.A[i,j]*prev_param
                    params.data.copy_(weighted_sum)
            #step2: x=x-lr*v
            for i,model in enumerate(self.model_list):
                for param,v in zip(model.parameters(),self.v_list[i]):
                    if v is not None:
                        update=self.lr*v
                        param.data -= update
        #step3: compute g
        for model in self.model_list:
            model.zero_grad()
            loss=closure()#自动完成了反向传播
        with torch.no_grad():
            #step4: v=Av+W@g-prev_W@prev_g
            new_v_list=[]
            for i,model in enumerate(self.model_list):
                new_v=[]
                for idx,(param,prev_param) in enumerate(zip(model.parameters(),self.prev_model_list[i].parameters())):
                    if param.grad is not None and prev_param.grad is not None:
                        weighted_v_sum=torch.zeros_like(param.grad)
                        for j in range(len(self.model_list)):
                            weighted_v_sum+=self.A[i,j]*self.v_list[j][idx]
                        v_update=weighted_v_sum+(1/self.W[i][i])*param.grad-(1/self.prev_W[i][i])*prev_param.grad
                        new_v.append(v_update.clone())
                    else:
                        new_v.append(None)
                new_v_list.append(new_v)
            self.v_list=new_v_list
            #step5
            self.prev_W=self.W
            for i,model in enumerate(self.model_list):
                self.prev_model_list[i].load_state_dict(model.state_dict())
                for prev_param,param in zip(self.prev_model_list[i].parameters(), model.parameters()):
                    if param.grad is not None:
                        prev_param.grad=param.grad.clone()
            self.prev_v_list = [[v.clone() if v is not None else None for v in self.v_list[i]] for i in range(len(self.model_list))]

        return loss