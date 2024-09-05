import torch
import time
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import utils
from pmw.weight_parallelized_subdomain import WeightParallelizedSubdomain
from pmw.weight_parallelized_tensor import WeightParallelizedTensor
from pmw.base_model import BaseModel

class WeightParallelizedModel(BaseModel):
    def __init__(self, stage_list, rank_list, sample):        
        '''
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the stage_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        ''' 
        super().__init__()
        self.rank_list = rank_list
        for k, ranks in enumerate(self.rank_list):
            if self.rank in ranks:
                self.layer_index = k
                break
        self.master_group = dist.new_group(ranks=utils.list_flattener(self.rank_list), use_local_synchronization=True)
        # (layer_class, params) = stage_list[self.layer_index]
        self.unbuilt_stage = stage_list[self.layer_index]# layer_class(**params).to(self.tensor_device) # Initialize the layer with provided parameters
        previous_layer_rank = None if self.layer_index-1 < 0 else self.rank_list[self.layer_index-1][0]
        next_layer_rank = None if self.layer_index+1 >= len(self.rank_list) else self.rank_list[self.layer_index+1][0]
        self.subdomain = WeightParallelizedSubdomain(previous_layer_rank, next_layer_rank, self.unbuilt_stage, sharded_on_ranks = ranks) # Initialize the subdomain model
        self.do_setup_phase(rank_list, sample)
        
    def do_setup_phase(self,rank_list, sample):
        self.setup_phase = True
        loss = None
        out = self.forward(torch.randn(*sample.shape).to(self.tensor_device), chunks_amount=1, reset_grad=True, compute_grad=True)
        if self.rank in rank_list[-1]:
            loss = nn.MSELoss()(out[0], torch.rand_like(out[0]))
        self.backward([loss])
        self.zero_grad()
        self.setup_phase = False

    def zero_grad(self):
        self.subdomain.zero_grad()
    
    def grad(self, clone=False): # Returns the global gradient of the model
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return WeightParallelizedTensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        params = [param.clone() if clone else param for param in self.subdomain.parameters()]
        return WeightParallelizedTensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2): # Returns the subdomain gradient norm of the model
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def forward(self, x, chunks_amount=1, reset_grad=False, compute_grad=True):
        # Initialize the input and output tensors on the subdomain (needed for the backward pass and to process data on their own)
        self.subdomain.inputs = [None]*chunks_amount
        self.subdomain.outputs = [None]*chunks_amount  
        
        compute_grad = False if not torch.is_grad_enabled() else compute_grad # flag to avoid storing the tensors needed to compute the gradients
        with torch.set_grad_enabled(compute_grad):
            if reset_grad:
                self.zero_grad() # Reset the gradients of the model before starting to accumulate them again
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32)
            if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = list(x.chunk(chunks_amount)) # Chunkenize the input tensor
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32) # Store the batch size of each chunk
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            chunk_shapes = chunk_shapes.to(self.backend_device(x)) # NOTE: Necessary for broadcast to work correctly, as to can be asynchronous when transferring to CUDA devices. Broadcasting only the batch size | async operation to avoid blocking the first layer
            dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0][0], group=self.master_group, async_op=False) # broadcasting only the batch size | async operation to avoid blocking the first layer
            # Go through the pipeline
            for c in range(chunks_amount):
                if self.layer_index == 0: # Beginning of the pipeline (first layer)
                    chunk = chunks[c].to(self.tensor_device)
                    self.subdomain.forward(chunk, is_in_pipeline = True, setup_phase=self.setup_phase, chunk_shapes=None) 
                else:
                    self.subdomain.forward(x = None, is_in_pipeline = True, setup_phase=self.setup_phase, chunk_shapes=chunk_shapes[c]) 

        return self.subdomain.outputs if self.rank in self.rank_list[-1] else [True]

    def backward(self, losses):
        self.subdomain.grad_outputs = [None for _ in range(len(losses))] # len(losses) is the chunks_amount
        counter = 0
        for loss in losses: # Chunked loss
            # print(f"Rank {self.rank} going through loss counter {counter}" )
            counter += 1
            self.subdomain.backward(loss, is_in_pipeline = True)
            # if self.rank in self.rank_list[-1]: # End of the pipeline
            #     grad_output = autograd.grad(loss, self.subdomain.outputs[c], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
            #     grad_data = autograd.grad(self.subdomain.outputs[c], self.subdomain.inputs[c], grad_outputs=grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
            #     dist.send(tensor=grad_data.to(self.backend_device(grad_data)), dst=self.rank_list[-2][0]) # TODO make this async if possible
                
            # elif self.rank in self.rank_list[0]: # Beginning of the pipeline
            #     grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
            #     dist.recv(grad_output, src=self.rank_list[1][0])       

            #     grad_output = grad_output.to(self.tensor_device).detach()
            # else: # Middle of the pipeline
            #     grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
            #     dist.recv(grad_output, src=self.rank_list[self.layer_index+1][0])       

            #     grad_output = grad_output.to(self.tensor_device).detach()
            #     grad_data = autograd.grad(self.subdomain.outputs[c], self.subdomain.inputs[c], grad_outputs=grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
            #     dist.send(tensor=grad_data.to(self.backend_device(grad_data)), dst=self.rank_list[self.layer_index-1][0]) # TODO make this async if possible
                
            
 