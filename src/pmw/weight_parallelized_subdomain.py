import torch
from pmw.base_model import BaseModel
from pmw.sharded_layer import ShardedLayer
from torch import nn
from torch import autograd
import utils
import torch.distributed as dist

class WeightParallelizedSubdomain(BaseModel):
    def __init__(self, previous_layer_rank, next_layer_rank, unbuilt_stage, sharded_on_ranks):
        super().__init__()
        self.unbuilt_stage = unbuilt_stage
        self.previous_layer_rank = previous_layer_rank
        self.next_layer_rank = next_layer_rank
        self.sharded_on_ranks = sharded_on_ranks
        self.outputs = []
        self.inputs = []
        self.grad_outputs = []
        if len(self.sharded_on_ranks) > 1:
            self.sharded_layers = [] # TODO: preallocate memory
            for layer, layer_settings in zip(self.unbuilt_stage[0], self.unbuilt_stage[1]):
                self.sharded_layers.append(ShardedLayer(layer=layer, layer_settings=layer_settings))
        else: # No sharding
            self.sharded_layers = nn.Sequential(*[layer(**layer_settings).to(self.tensor_device) for layer, layer_settings in zip(self.unbuilt_stage[0], self.unbuilt_stage[1])])
    
    def forward(self, x=None, is_in_pipeline=False, setup_phase=False, chunk_shapes=None):
        if x is None and not is_in_pipeline:
            for k, chunk in enumerate(self.inputs):
                for layer in self.sharded_layers:
                    chunk = layer.forward(chunk)
                self.outputs[k] = chunk
            return self.outputs
        else:
            if is_in_pipeline:
                if self.previous_layer_rank is not None:
                    if setup_phase:
                        shapes = utils.receive_shape(src=self.previous_layer_rank, device=self.backend_device())
                        x = torch.empty(*shapes, device=self.backend_device(), requires_grad=True)
                    else:
                        x = torch.empty(*self.shapes[0](chunk_shapes), device=self.backend_device(), requires_grad=True)
                    dist.recv(tensor=x, src=self.previous_layer_rank)
                x = x.to(self.tensor_device)
                    
            # Index of the first None in list self.inputs (current chunk), else -1
            k = self.inputs.index(None) if None in self.inputs else len(self.inputs) - 1          
            if k == -1: # Nothing to store
                for layer in self.sharded_layers:
                    x = layer.forward(x)
            else:
                self.inputs[k] = x
                for layer in self.sharded_layers:
                    x = layer.forward(x)
                self.outputs[k] = x
                
            if is_in_pipeline:
                if setup_phase:
                    input_shape = lambda z: [z]+list(shapes)[1:]
                    output_shape = lambda z: [z]+list(self.outputs[k].shape)[1:]
                    self.shapes = [input_shape, output_shape]
                    if self.next_layer_rank is not None:
                        utils.send_shape(self.outputs[k].shape, dst=self.next_layer_rank, device=self.backend_device(self.outputs[k]))
                if self.next_layer_rank is not None:
                    self.outputs[k] = self.outputs[k].to(self.backend_device(self.outputs[k]))
                    dist.send(tensor=self.outputs[k], dst=self.next_layer_rank) # send the tensor
                    self.outputs[k] = self.outputs[k].to(self.tensor_device)
            return x
        
    def backward(self, loss=None, is_in_pipeline=False):
        if is_in_pipeline:
            k = self.grad_outputs.index(None) if None in self.grad_outputs else len(self.grad_outputs) - 1
            loop = [k]
            chunk_size = self.outputs[k].shape[0]
            if self.next_layer_rank is None: # End of the pipeline
                grad_output = autograd.grad(loss, self.outputs[k], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
            else: 
                grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
                if self.next_layer_rank is not None:
                    dist.recv(grad_output, src=self.next_layer_rank)       
                grad_output = grad_output.to(self.tensor_device).detach()
            if self.previous_layer_rank is not None:
                grad_data = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                grad_data = grad_data.to(self.backend_device(grad_data))
                dist.send(tensor=grad_data, dst=self.previous_layer_rank)
            self.grad_outputs[k] = grad_output
        else:
            loop = range(len(self.outputs))
            
        for k in loop:
            if len(self.sharded_on_ranks) > 1:
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
