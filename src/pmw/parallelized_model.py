import torch
import torch.nn as nn
import torch.distributed as dist
import utils
from pmw.base_model import BaseModel
from pmw.weight_parallelized_model import WeightParallelizedModel
from pmw.data_and_weight_parallelized_subdomain import DataAndWeightParallelizedSubdomain


class ParallelizedModel(BaseModel):
    def __init__(self, stage_list, sample, num_replicas_per_subdomain=1, num_subdomains=1):
        '''
        Ranks that will be used are [0, ..., world_size - 1]

        This is the outermost shell in a multi-shell parallel strategy.
        num_subdomains is the next shell, which deals with data-parallelism (Domain Decomposition approach).
        num_replicas_per_subdomain refers to the number of replicas in each data-parallel subdomain (exact data parallelism to speed up computation within each subdomain).
        stage_list is the list of pipeline stages per replica in each subdomain

        E.g. num_subdomains = 2; num_replicas_per_subdomain = 3; stage_list = [(Layer0, Layer0dict), (Layer1, Layer1dict), (Layer2, Layer2dict)])
    
                                    Subdomain 0                                                         Subdomain 1
                Replica 0           Replica 1           Replica 2                   Replica 0           Replica 1           Replica 2
                [Layer0, (Rank0)    [Layer0, (Rank3)   [Layer0, (Rank6)            [Layer0, (Rank9)    [Layer0, (Rank12)   [Layer0, (Rank15)
                 Layer1, (Rank1)     Layer1, (Rank4)    Layer1, (Rank7)             Layer1, (Rank10)    Layer1, (Rank13)    Layer1, (Rank16)
                 Layer2] (Rank2)     Layer2] (Rank5)    Layer2] (Rank8)             Layer2] (Rank11)    Layer2] (Rank14)    Layer2] (Rank17)
        '''
        super().__init__()
        if num_replicas_per_subdomain*num_subdomains*len(stage_list) != dist.get_world_size():
            raise ValueError("The number of replicas per subdomain times the number of subdomains times the length of the stage_list must be equal to the world size.")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.stage_list = stage_list # Model copy rank list
        self.num_subdomains = num_subdomains
        self.num_replicas_per_subdomain = num_replicas_per_subdomain
        self.tot_replicas = num_subdomains * num_replicas_per_subdomain
    
        # Model ranks is a list of lists, where each list contains the rank corresponding to a model replica
        self.all_model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(self.tot_replicas)] 
        # Split self.model_ranks into num_subdomains pieces
        self.all_model_ranks_per_subdomain = [self.all_model_ranks[k:k+num_replicas_per_subdomain] for k in range(0, len(self.all_model_ranks), num_replicas_per_subdomain)]
        
        for model_rank_list in self.all_model_ranks_per_subdomain:
            model_rank_flat_list = [item for sublist in model_rank_list for item in sublist]
            if self.rank in model_rank_flat_list:
                self.subdomain = DataAndWeightParallelizedSubdomain(stage_list, model_rank_list, sample, num_replicas_per_subdomain)
                break

        self._create_process_groups()
        self.sync_params()
        
    def _create_process_groups(self):
        # TODO: Avoid local sync before the global one
        # NOTE: if we only have one subdomain, sync should be done on the subdomain and not globally
        self.layer_copies_group = None
        if self.tot_replicas > 1:
            use_local_synchronization = True
            self.all_stage_ranks = [[self.all_model_ranks[i][j] for i in range(self.tot_replicas)] for j in range(len(self.stage_list))]
            for stage_ranks in self.all_stage_ranks:
                if self.rank in stage_ranks:
                    self.layer_copies_group = dist.new_group(ranks=stage_ranks, use_local_synchronization=use_local_synchronization)
                    break

    def subdomain_forward(self):
        return self.subdomain.weight_parallelized_model.subdomain.forward()

    def subdomain_backward(self, sync=True):
        self.subdomain.weight_parallelized_model.subdomain.backward()

    def subdomain_params(self):
        return self.subdomain.weight_parallelized_model.subdomain.parameters()
        
    def subdomain_grad(self):
        return self.subdomain.weight_parallelized_model.subdomain.grad()
    
    def subdomain_grad_norm(self, p=2):
        return self.subdomain.weight_parallelized_model.subdomain.grad_norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        return self.subdomain.weight_parallelized_model.parameters(clone=clone)
    
    def parameters_norm(self, p=2): # Returns the global parameters norm of the model
        return self.subdomain.weight_parallelized_model.parameters().norm(p=p)
    
    def grad(self, clone=False): # Returns the global gradient of the model
        return self.subdomain.weight_parallelized_model.grad(clone=clone)

    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.subdomain.weight_parallelized_model.grad_norm(p=p)
        
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.subdomain.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses):
        self.subdomain.backward(losses=losses)
        self.sync_grads()  
    
    def sync_params(self, method='average'):
        if self.num_subdomains > 1:
            for param in self.subdomain_params():
                param.data = param.data.to(self.subdomain.backend_device(param.data))
                dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.data = param.data.to('cuda') # TODO: Take into account tensor sharding
                if method == 'average':
                    param.data /= self.num_replicas
                elif method == 'sum':
                    pass # nothing to do since we already summed the parameters through all_reduce
                else:
                    raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        if self.num_subdomains > 1:
            for param in self.subdomain_params():
                dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.grad /= self.num_replicas

# class ParallelizedModel(nn.Module):
#     def __init__(self, stage_list, sample, num_replicas=1, device=None, data_parallel=False):
#         super(ParallelizedModel, self).__init__()
#         if num_replicas <= 1 and data_parallel:
#             raise ValueError("Data parallelism requires at least two replicas.")

#         self.num_replicas = num_replicas
#         self.world_size = dist.get_world_size()
#         self.data_parallel = data_parallel
#         self.backend_device = 'cpu' if dist.get_backend() == 'gloo' else 'cuda:0' # TODO: remove if not used
#         self.tensor_device = utils.decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
#         if num_replicas*len(stage_list) != self.world_size:
#             raise ValueError(f"The number of replicas times the number of layers ({num_replicas}*{len(stage_list)}={num_replicas*len(stage_list)}) must be equal to the world size ({self.world_size}).")
#         self.rank = dist.get_rank()
#         self.final_layer_group = dist.new_group(ranks=[r for r in range(self.world_size) if r%len(stage_list) == len(stage_list)-1], use_local_synchronization=True)

#         # Model copy rank list
#         self.stage_list = stage_list
#         self.model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(num_replicas)] 
#         for model_rank in self.model_ranks:
#             if self.rank in model_rank:
#                 self.rank_list = model_rank
#                 self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
#                 break
#         for ranks in self.model_ranks:
#             if self.rank in ranks:
#                 # TODO: Change gpu_id to be more generic for tensor sharding later on...
#                 self.model = WeightParallelizedModel(stage_list=stage_list, rank_list=ranks, sample=sample)
#                 self.subdomain = self.model.subdomain
        
#         # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
#         for layer_idx in range(len(self.stage_list)):
#             # Collect ranks that are responsible for this layer across all replicas
#             ranks = [r + layer_idx for r in range(0, self.world_size, len(self.stage_list))]
#             # Create a new group containing these ranks
#             if self.rank in ranks:
#                 self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
#         self.sync_params()
    
#     def subdomain_forward(self, sync=False):
#         outputs = self.subdomain.forward()
#         if sync: # NOTE: Probably never used, but included for completeness
#             with torch.no_grad():
#                 for output in outputs:
#                     dist.all_reduce(output, group=self.final_layer_group, op=dist.ReduceOp.SUM)
#         return outputs
    
#     def subdomain_backward(self, sync=True):
#         self.subdomain.backward()
#         if sync and self.data_parallel: # NOTE: Sync True for sure for data parallel settings, but leaving the option to the user to use False
#             self.sync_grads() 

#     def subdomain_params(self):
#         return self.subdomain.parameters()
        
#     def subdomain_grad(self):
#         return self.subdomain.grad()
    
#     def subdomain_grad_norm(self, p=2):
#         return self.subdomain.grad_norm(p=p)
    
#     def parameters(self, clone=False): # Returns the global parameters of the model
#         return self.model.parameters(clone=clone)
    
#     def parameters_norm(self, p=2): # Returns the global parameters norm of the model
#         return self.model.parameters().norm(p=p)
    
#     def grad(self, clone=False): # Returns the global gradient of the model
#         return self.model.grad(clone=clone)

#     def grad_norm(self, p=2): # Returns the global gradient norm of the model
#         return self.model.grad_norm(p=p)
        
#     def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
#         return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
#     def backward(self, losses, sync=True):
#         self.model.backward(losses=losses)
#         if sync: # Synchronize the gradients across all replicas (True by default since this will always be done in both data parallel approaches)
#             self.sync_grads() 
    
#     def sync_params(self, method='average'):
#         for param in self.model.parameters():
#             dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
#             if method == 'average':
#                 param.data /= self.num_replicas
#             elif method == 'sum':
#                 pass # nothing to do since we already summed the parameters through all_reduce
#             else:
#                 raise ValueError(f"Method {method} is not supported.")

#     def sync_grads(self):
#         for param in self.model.parameters():
#             dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
#             param.grad /= self.num_replicas