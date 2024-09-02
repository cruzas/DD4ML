import torch
from pmw.base_model import BaseModel
from pmw.sharded_layer import ShardedLayer
from torch import nn
from torch import autograd

class WeightParallelizedSubdomain(BaseModel):
    def __init__(self, unbuilt_stage, is_sharded:bool = True):
        super().__init__()
        self.unbuilt_stage = unbuilt_stage
        self.outputs = []
        self.inputs = []
        self.grad_outputs = []
        self.is_sharded = is_sharded
        if self.is_sharded: # TODO: verify
            self.sharded_layers = [] # TODO: preallocate memory
            for layer, layer_settings in zip(self.unbuilt_stage[0], self.unbuilt_stage[1]):
                self.sharded_layers.append(ShardedLayer(layer=layer, layer_settings=layer_settings, is_sharded=is_sharded))
        else:
            self.sharded_layers = nn.Sequential(*[layer(**layer_settings).to(self.tensor_device) for layer, layer_settings in zip(self.unbuilt_stage[0], self.unbuilt_stage[1])])
    
    def forward(self, x=None):
        try:
            print(f"Rank {self.rank}, subdomain x shape: {x.shape}")
        except:
            pass
        if x is None:
            for k, chunk in enumerate(self.inputs):
                for layer in self.sharded_layers:
                    chunk = layer.forward(chunk)
                self.outputs[k] = chunk
            print(f"Rank {self.rank}, case 1 num outputs: {len(self.outputs)}, outputs shape: {self.outputs[0].shape}")
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
                for layer in self.sharded_layers:
                    x = layer.forward(x)
                self.outputs[k] = x
                print(f"Rank {self.rank}, case 2 num outputs: {len(self.outputs)}, outputs shape: {self.outputs[k].shape}")
                return self.outputs[k]
        
    def backward(self, grad_output=None):
        if grad_output is None:
            loop = range(len(self.outputs))
        else:
            k = self.grad_outputs.index(None) if None in self.grad_outputs else len(self.grad_outputs) - 1
            loop = [k]
            self.grad_outputs[k] = grad_output
        for k in loop:
            if self.is_sharded:
                for layer in reversed(self.sharded_layers):
                    layer.backward(self.outputs[k], self.grad_outputs[k], len(self.outputs))
            else:
                for param in self.parameters():
                    grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_outputs[k], retain_graph=True)[0] / len(self.outputs)
                    param.grad = grad if param.grad is None else param.grad + grad

    def grad(self):
        # TODO: Implement sharded_layers.parameters()
        return [param.grad for param in self.sharded_layers.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.sharded_layers.parameters()], dim=0), p=2).item()
