import time 
import torch
import torch.nn as nn
import torch.autograd as autograd
from pmw.model import BaseModel
from pmw.sharded_layer import ShardedLayer

class WeightParallelizedSubdomain(BaseModel):
    def __init__(self, stage, is_sharded:bool = False):
        super().__init__()
        self.stage = stage
        self.outputs = []
        self.inputs = []
        self.grad_output = []
        self.is_sharded = is_sharded
        if self.is_sharded: # TODO: verify
            self.sharded_layers = [] # TODO: preallocate memory
            for layer in self.stage:
                self.sharded_layers.append(ShardedLayer(layer=layer, is_sharded=is_sharded))
    
    def forward(self, x=None):
        if x is None:
            for k, chunk in enumerate(self.inputs):
                self.outputs[k] = self.stage(chunk)
            return self.outputs
        else:
            # Index of the first None in list self.inputs (current chunk), else -1
            k = self.inputs.index(None) if None in self.inputs else len(self.inputs) - 1          
            if k == -1: # Nothing to store
                return self.stage(x)
            else:
                self.inputs[k] = x
                self.outputs[k] = self.stage(x)
                return self.outputs[k]
        
    def backward(self, grad_output=None):
        if grad_output is None:
            loop = range(len(self.outputs))
        else:
            k = self.grad_output.index(None) if None in self.grad_output else len(self.grad_output) - 1
            loop = [k]
            self.grad_output[k] = grad_output
        for k in loop:
            for param in self.parameters():
                grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0] / len(self.outputs)
                param.grad = grad if param.grad is None else param.grad + grad

    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()
