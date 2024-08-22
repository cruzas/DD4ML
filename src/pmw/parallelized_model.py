import torch
import torch.nn as nn
import torch.distributed as dist
from utils import Utils
from pmw.weight_parallelized_model import WeightParallelizedModel

class ParallelizedModel(nn.Module):
    def __init__(self, stage_list, sample, num_replicas=1, device=None, data_parallel=False):
        super(ParallelizedModel, self).__init__()
        if num_replicas <= 1 and data_parallel:
            raise ValueError("Data parallelism requires at least two replicas.")

        self.num_replicas = num_replicas
        self.world_size = dist.get_world_size()
        self.data_parallel = data_parallel
        # self.backend_device = 'cpu' if dist.get_backend() == 'gloo' else 'cuda:0' # TODO: remove if not used
        self.tensor_device = Utils.decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        if num_replicas*len(stage_list) != self.world_size:
            raise ValueError(f"The number of replicas times the number of layers ({num_replicas}*{len(stage_list)}={num_replicas*len(stage_list)}) must be equal to the world size ({self.world_size}).")
        self.rank = dist.get_rank()
        self.final_layer_group = dist.new_group(ranks=[r for r in range(self.world_size) if r%len(stage_list) == len(stage_list)-1], use_local_synchronization=True)

        # Model copy rank list
        self.stage_list = stage_list
        self.model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(num_replicas)] 
        for model_rank in self.model_ranks:
            if self.rank in model_rank:
                self.rank_list = model_rank
                self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
                break
        for ranks in self.model_ranks:
            if self.rank in ranks:
                # TODO: Change gpu_id to be more generic for tensor sharding later on...
                self.model = WeightParallelizedModel(stage_list=stage_list, rank_list=ranks, sample=sample, device=device)
                self.subdomain = self.model.subdomain
        
        # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
        for layer_idx in range(len(self.stage_list)):
            # Collect ranks that are responsible for this layer across all replicas
            ranks = [r + layer_idx for r in range(0, self.world_size, len(self.stage_list))]
            # Create a new group containing these ranks
            if self.rank in ranks:
                self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
        self.sync_params()
    
    def subdomain_forward(self, sync=False):
        outputs = self.subdomain.forward()
        if sync: # NOTE: Probably never used, but included for completeness
            with torch.no_grad():
                for output in outputs:
                    dist.all_reduce(output, group=self.final_layer_group, op=dist.ReduceOp.SUM)
        return outputs
    
    def subdomain_backward(self, sync=True):
        self.subdomain.backward()
        if sync and self.data_parallel: # NOTE: Sync True for sure for data parallel settings, but leaving the option to the user to use False
            self.sync_grads() 

    def subdomain_params(self):
        return self.subdomain.parameters()
        
    def subdomain_grad(self):
        return self.subdomain.grad()
    
    def subdomain_grad_norm(self, p=2):
        return self.subdomain.grad_norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        return self.model.parameters(clone=clone)
    
    def parameters_norm(self, p=2): # Returns the global parameters norm of the model
        return self.model.parameters().norm(p=p)
    
    def grad(self, clone=False): # Returns the global gradient of the model
        return self.model.grad(clone=clone)

    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.model.grad_norm(p=p)
        
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses, sync=True):
        self.model.backward(losses=losses)
        if sync: # Synchronize the gradients across all replicas (True by default since this will always be done in both data parallel approaches)
            self.sync_grads() 
    
    def sync_params(self, method='average'):
        for param in self.model.parameters():
            dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            if method == 'average':
                param.data /= self.num_replicas
            elif method == 'sum':
                pass # nothing to do since we already summed the parameters through all_reduce
            else:
                raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        for param in self.model.parameters():
            dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            param.grad /= self.num_replicas