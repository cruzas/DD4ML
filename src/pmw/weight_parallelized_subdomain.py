import torch
from pmw.base_model import BaseModel
from pmw.sharded_layer import ShardedLayer

class WeightParallelizedSubdomain(BaseModel):
    def __init__(self, stage, is_sharded:bool = True):
        super().__init__()
        self.stage = stage
        self.outputs = []
        self.inputs = []
        self.grad_outputs = []
        self.is_sharded = is_sharded
        if self.is_sharded: # TODO: verify
            self.sharded_layers = [] # TODO: preallocate memory
            for layer in self.stage:
                self.sharded_layers.append(ShardedLayer(layer=layer, is_sharded=is_sharded))
        else:
            self.sharded_layers = self.stage
    
    def forward(self, x=None):
        if x is None:
            for k, chunk in enumerate(self.inputs):
                for layer in self.sharded_layers:
                    chunk = layer.forward(chunk)
                self.outputs[k] = chunk
                # self.outputs[k] = self.stage(chunk)
            return self.outputs
        else:
            # Index of the first None in list self.inputs (current chunk), else -1
            k = self.inputs.index(None) if None in self.inputs else len(self.inputs) - 1          
            if k == -1: # Nothing to store
                for layer in self.sharded_layers:
                    x = layer.forward(x)
                return x
            else:
                self.inputs[k] = x
                self.outputs[k] = self.stage(x)
                return self.outputs[k]
        
    def backward(self, grad_output=None):
        if grad_output is None:
            loop = range(len(self.outputs))
        else:
            k = self.grad_outputs.index(None) if None in self.grad_outputs else len(self.grad_outputs) - 1
            loop = [k]
            self.grad_outputs[k] = grad_output
        for k in loop:
            for layer in reversed(self.sharded_layers):
                layer.backward(self.outputs[k], self.grad_outputs[k], len(self.outputs))

    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()
