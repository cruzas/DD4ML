import torch
from pmw.base_model import BaseModel
from pmw.sharded_layer import ShardedLayer
from torch import nn
from torch import autograd
import utils
import torch.distributed as dist

class WeightParallelizedSubdomain(BaseModel):
    # This corresponds to a STAGE in a pipelined model.
    def __init__(self, model_handler):
        super().__init__()
        self.model_handler = model_handler
        self.sd, self.rep, self.s = self.model_handler.rank_to_sd_rep_s()

        self.outputs = []
        self.inputs = {} # key is te name of the layer we want to receive the input from, value is the tensor
        self.grad_outputs = []
        self.shapes = {}
        self.local_communication = {}
        
        self.stage_data = model_handler.stage_data()
        self.sharded_layers = []

        if self.rank in self.stage_data['ranks']:
            for layer_name in self.stage_data['layers']:
                self.sharded_layers.append(ShardedLayer(layer_dict=self.model_handler.net_dict[layer_name], layer_ranks=self.stage_data['ranks']))
    
    def forward(self, num_chunks, num_samples_in_chunk, chunk_id, x=None, is_in_pipeline=False, setup_phase=False):
        if chunk_id == 0:
            self.inputs = {}

        if x is None and not is_in_pipeline:
            for k, chunk in enumerate(self.inputs):
                for layer in self.sharded_layers:
                    chunk = layer.forward(chunk)
                self.outputs[k] = chunk
            return self.outputs
        else: # here we are in a pipeline and x is not None just for the first stage
            # Write a list of receive function to get the needed tensors (async), store such information in self.inputs with correct keys. Set a "wait" before using received tensors.]
            for layer_name in self.stage_data['layers']:
                for src_name in self.model_handler.net_dict[layer_name]['rcv']['src']:
                    current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                    src_layer_stage = self.model_handler.net_dict[src_name]['stage']
                    if current_layer_stage != src_layer_stage and src_name not in self.inputs.keys():
                        src_ranks = self.model_handler.layer_name_to_ranks(src_name) 
                        src_rank = src_ranks[0] # TODO: maybe improve send/rcv when tensor sharding is implemented
                        if setup_phase:
                            rcv_shape = utils.receive_shape(src=src_name, device=self.backend_device())
                            self.shapes[src_name] = lambda z: [z]+list(rcv_shape)[1:]
                        temp = torch.empty(*self.shapes[src_name](num_samples_in_chunk), device=self.backend_device(), requires_grad=True)
                        dist.recv(src=src_rank, tensor=temp) # TODO: make it async to speed up communication
                        self.inputs[src_name][chunk_id] = temp
                    
            for layer_name in self.stage_data['layers']:
                pass 
            # Here write local_communication var
            # out = ...

            # DO SEND
            for layer_name in self.stage_data['layers']:
                for dst_name in self.model_handler.net_dict[layer_name]['dst']['to']:
                    if dst_name not in self.outputs.keys(): # trying to receive the same tensor twice
                        dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                        dst_rank = dst_ranks[0] # TODO: maybe improve send/rcv when tensor sharding is implemented

                        current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                        dst_layer_stage = self.model_handler.net_dict[dst_name]['stage']
                        if current_layer_stage != dst_layer_stage:
                            if setup_phase:
                                utils.send_shape(to=dst_name, device=self.backend_device())
                                self.shapes[src_name] = lambda z: [z]+list(send_shape)[1:]
                            temp = torch.empty(*self.shapes[src_name](num_samples_in_chunk), device=self.backend_device(), requires_grad=True)
                            dist.send(to=dst_rank, tensor=temp) # TODO: make it async to speed up communication
                            self.inputs[src_name][chunk_id] = temp
                        else:
                            self.local_communication[src_name] = 0
                        

            #     if self.previous_layer_rank is not None:
            #         if setup_phase:
            #             shapes = utils.receive_shape(src=self.previous_layer_rank, device=self.backend_device())
            #             x = torch.empty(*shapes, device=self.backend_device(), requires_grad=True)
            #         else:
            #             x = torch.empty(*self.shapes[0](chunk_shapes), device=self.backend_device(), requires_grad=True)
            #         dist.recv(tensor=x, src=self.previous_layer_rank)
            #     x = x.to(self.tensor_device)
                    
            # # Index of the first None in list self.inputs (current chunk), else -1
            # k = self.inputs.index(None) if None in self.inputs else len(self.inputs) - 1          
            # if k == -1: # Nothing to store
            #     for layer in self.sharded_layers:
            #         x = layer.forward(x)
            # else:
            #     self.inputs[k] = x
            #     for layer in self.sharded_layers:
            #         x = layer.forward(x)
            #     self.outputs[k] = x # TODO: add skip connection value to x, if any
                
            # if is_in_pipeline:
            #     if setup_phase:
            #         input_shape = lambda z: [z]+list(shapes)[1:]
            #         output_shape = lambda z: [z]+list(self.outputs[k].shape)[1:]
            #         self.shapes = [input_shape, output_shape]
            #         if self.next_layer_rank is not None:
            #             utils.send_shape(self.outputs[k].shape, dst=self.next_layer_rank, device=self.backend_device(self.outputs[k]))
            #     if self.next_layer_rank is not None:
            #         self.outputs[k] = self.outputs[k].to(self.backend_device(self.outputs[k]))
            #         dist.send(tensor=self.outputs[k], dst=self.next_layer_rank) # send the tensor
            #         self.outputs[k] = self.outputs[k].to(self.tensor_device)
            # return x
        
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
