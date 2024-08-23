import time 
import torch
import torch.nn as nn
import torch.autograd as autograd
from pmw.model import BaseModel

class WeightParallelizedSubdomain(BaseModel):
    def __init__(self, model):
        super(WeightParallelizedSubdomain, self).__init__()
        self.model = model
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations   
        self.f_time = 0
        self.g_time = 0
        
    def forward(self):
        start = time.time()
        for k, chunk in enumerate(self.model.inputs):
            self.model.outputs[k] = self.model.stage(chunk)
        self.num_f_evals += 1
        self.f_time += time.time() - start
        return self.model.outputs

    def backward(self):
        start = time.time()
        for k in range(len(self.model.outputs)):
            for param in self.model.stage.parameters():
                if param.grad is None:
                    param.grad = autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                else:
                    param.grad += autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
        self.num_g_evals += 1
        self.g_time += time.time() - start
            
    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()
