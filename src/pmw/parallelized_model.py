import torch
import torch.nn as nn
import torch.distributed as dist
import utils
from pmw.base_model import BaseModel
from pmw.weight_parallelized_model import WeightParallelizedModel
from pmw.data_and_weight_parallelized_subdomain import DataAndWeightParallelizedSubdomain


class ParallelizedModel(BaseModel):
    def __init__(self, stage_list, sample, num_replicas_per_subdomain=1, num_subdomains=1, is_sharded: bool = False):
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
        if num_replicas_per_subdomain*num_subdomains*len(stage_list) < self.world_size:
            # print("Num replicas per subdomain: ", num_replicas_per_subdomain)
            # print("Num subdomains: ", num_subdomains)
            # print("Length of stage list: ", len(stage_list))
            # print("The three multiplied together: ", num_replicas_per_subdomain*num_subdomains*len(stage_list))
            # print("World size: ", dist.get_world_size())
            raise ValueError(f"The number of replicas per subdomain {num_replicas_per_subdomain} times the number of subdomains {num_subdomains} times the length of the stage_list {len(stage_list)} (= {num_replicas_per_subdomain*num_subdomains*len(stage_list)}) must be equal or larger to the world size ({self.world_size}).")

        self.stage_list = stage_list # Model copy rank list
        self.num_subdomains = num_subdomains
        self.num_replicas_per_subdomain = num_replicas_per_subdomain
        self.tot_replicas = num_subdomains * num_replicas_per_subdomain
        self.is_sharded = is_sharded
        self.node_rank_dict = utils.gather_node_info() # e.g. {'node1': [0, 1, 2,3], 'node2': [4, 5,6,7], 'node3': [8,9,10,11], 'node4': [12,13,14,15], 'node5': [16,17,18,19], 'node6': [20,21,22,23]} -> each node will have 4 gpus with global rank numbers within the list
    
        # Model ranks is a list of lists, where each list contains the rank corresponding to a model replica
        self.distributed_model_rank_structure(num_subdomains, num_replicas_per_subdomain, len(stage_list), 1, self.node_rank_dict)
        # self.all_model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(self.tot_replicas)] 
        # Split self.model_ranks into num_subdomains pieces
        # self.all_model_ranks_per_subdomain = [self.all_model_ranks[k:k+num_replicas_per_subdomain] for k in range(0, len(self.all_model_ranks), num_replicas_per_subdomain)]

        # if the current rank is in "self.all_ranks_group" then run the rest of the code (Note: self.all_ranks_group is a dist.new_group)
        if self.rank in self.all_model_ranks_flat:
            self.subdomain = DataAndWeightParallelizedSubdomain(stage_list, self.subdomain_rank_structure(), sample)
            self.sync_params()
    
    def parameters(self):
        return self.subdomain.weight_parallelized_model.subdomain.parameters()

    def subdomain_forward(self):
        return self.subdomain.weight_parallelized_model.subdomain.forward()

    def subdomain_backward(self):
        self.subdomain.weight_parallelized_model.subdomain.backward()
        self.subdomain.sync_grads() # This is needed in case there are multiple replicas per subdomain (exact data parallelism)

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
                dist.all_reduce(tensor=param.data, group=self.all_layer_copies_group, op=dist.ReduceOp.SUM)
                if method == 'average':
                    param.data /= self.tot_replicas
                elif method == 'sum':
                    pass # nothing to do since we already summed the parameters through all_reduce
                else:
                    raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        if self.num_subdomains > 1:
            for param in self.subdomain_params():
                dist.all_reduce(tensor=param.grad, group=self.all_layer_copies_group, op=dist.ReduceOp.SUM)	
                param.grad /= self.tot_replicas
