import torch
from pmw.base_model import BaseModel
from pmw.sharded_layer import ShardedLayer
from torch import nn
from torch import autograd
import utils
import torch.distributed as dist
import copy

class WeightParallelizedSubdomain(BaseModel):
    # This corresponds to a STAGE in a pipelined model.
    def __init__(self, model_handler):
        super().__init__()
        self.model_handler = model_handler
        self.sd, self.rep, self.s, self.sh = self.model_handler.sd, self.model_handler.rep, self.model_handler.s, self.model_handler.sh

        self.inputs = {}
        self.outputs = {}
        self.grad_outputs = {}
        self.shapes = {}
        
        self.stage_data = model_handler.stage_data()
        self.sharded_layers = []

        if self.rank in self.stage_data['ranks']:
            for layer_name in self.stage_data['layers']:
                self.sharded_layers.append(ShardedLayer(layer_dict=self.model_handler.net_dict[layer_name], layer_ranks=self.stage_data['ranks']))
    
    def parameters(self):  
        return [param for layer in self.sharded_layers for param in layer.parameters()]

    def forward(self, num_chunks, num_samples_in_chunk, chunk_id, x=None, is_in_pipeline=False, setup_phase=False):
        empty_at_the_end = []
        if x is None and not is_in_pipeline:
            for k, chunk in enumerate(self.inputs):
                for layer in self.sharded_layers:
                    chunk = layer.forward(chunk)
                self.outputs[k] = chunk
        else: # Here we are in a pipeline and x is not None just for the first stage
            for layer_name in self.stage_data['layers']:
                for src_name in self.model_handler.net_dict[layer_name]['rcv']['src']:
                    current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                    src_layer_stage = self.model_handler.net_dict[src_name]['stage']
                    key = layer_name+'_'+src_name
                    if current_layer_stage != src_layer_stage:
                        src_ranks = self.model_handler.layer_name_to_ranks(src_name) 
                        src_rank = src_ranks[0] # TODO: maybe improve send/rcv when tensor sharding is implemented
                        if setup_phase:
                            rcv_shape = utils.receive_shape(src=src_rank, device=self.backend_device())
                            self.shapes[key] = lambda z, temp_shape=copy.deepcopy(list(rcv_shape)[1:]): [z] + temp_shape
                        temp = torch.empty(*self.shapes[key](num_samples_in_chunk), device=self.backend_device(), requires_grad=True)
                        dist.recv(src=src_rank, tensor=temp) # TODO: make it async to speed up communication
                        
                        if key not in self.inputs.keys() or len(self.inputs[key]) != num_chunks:
                            self.inputs[key] = [None]*num_chunks
                        self.inputs[key][chunk_id] = temp.to(self.tensor_device)

            for i, layer_name in enumerate(self.stage_data['layers']):
                input_list = self.model_handler.net_dict[layer_name]['rcv']['src']
                strategy_in = self.model_handler.net_dict[layer_name]['rcv']['strategy']
                if layer_name != 'start':
                    if strategy_in is None:
                        x = self.inputs[layer_name+'_'+input_list[0]][chunk_id]
                    else:
                        x = strategy_in(*[self.inputs[layer_name+'_'+src_name][chunk_id] for src_name in input_list])
                out = self.sharded_layers[i].forward(x)
                if isinstance(out, list) and len(self.model_handler.net_dict[layer_name]['dst']['to']) != len(out):
                    raise ValueError(f"Output of layer {layer_name} is a list of torch.Tensor with length different from the number of destination layers")
                elif not isinstance(out, torch.Tensor) and not isinstance(out, list):
                    raise TypeError(f"Output of the callable object with label {layer_name} is of type {type(out)}. Only torch.Tensor or List (of torch.Tensor) is allowed.")
                
                if layer_name == 'finish': 
                    if chunk_id == 0:
                        self.outputs['finish'] = [None]*num_chunks
                    self.outputs['finish'][chunk_id] = temp.to(self.tensor_device) 

                for dst_idx, dst_name in enumerate(self.model_handler.net_dict[layer_name]['dst']['to']):
                    dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                    dst_rank = dst_ranks[0] # TODO: maybe improve send/rcv when tensor sharding is implemented
                    key = layer_name+'_'+dst_name 
                    current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                    dst_layer_stage = self.model_handler.net_dict[dst_name]['stage']
                    temp = out if not isinstance(out, list) else out[dst_idx]
                    if current_layer_stage != dst_layer_stage:
                        temp = temp.to(self.backend_device())
                        if setup_phase:
                            utils.send_shape(shape=temp.shape, dst=dst_rank, device=self.backend_device())
                        dist.send(tensor=temp, dst=dst_rank)
                        
                        if key not in self.outputs.keys() or len(self.outputs[key]) != num_chunks:
                            self.outputs[key] = [None]*num_chunks
                        self.outputs[key][chunk_id] = temp.to(self.tensor_device)  
                    else:
                        reverse_key = dst_name+'_'+layer_name
                        empty_at_the_end.append(reverse_key)
                        self.inputs[reverse_key] = [None]*num_chunks                         
                        self.inputs[reverse_key][chunk_id] = temp                 
            for key in empty_at_the_end:
                del self.inputs[key]
                self.inputs[key] = [None]*num_chunks
                
        return self.outputs if self.model_handler.is_last_stage() else [True]
        
    def backward(self, loss=None, num_chunks=1, chunk_id=0, is_in_pipeline=False):
        if is_in_pipeline:
            for layer_name in reversed(self.stage_data['layers']):
                for dst_idx, dst_name in enumerate(self.model_handler.net_dict[layer_name]['dst']['to']):
                    key = layer_name+'_'+dst_name
                    chunk_size = self.outputs[key][chunk_id].shape[0]
                    

            if self.next_layer_rank is None: # End of the pipeline
                grad_output = autograd.grad(loss, self.outputs[k], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
            else: 
                grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
                if self.next_layer_rank is not None:
                    dist.recv(grad_output, src=self.next_layer_rank)       
                grad_output = grad_output.to(self.tensor_device).detach()
            if self.previous_layer_rank is not None:
                grad_data = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=grad_output, retain_graph=True)[0] # This is needed to compute the derivative at the previous stage
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
