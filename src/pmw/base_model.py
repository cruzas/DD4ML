from abc import abstractmethod
import torch.nn as nn
import torch.distributed as dist
import torch
import utils


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.backend = dist.get_backend()
        self.tensor_device = torch.device('cuda')
        self.default_device = torch.device('cpu') # Default device to store scalars
    
    def backend_device(self, tensor=torch.tensor([0])):
        return torch.device('cpu') if dist.get_backend() == 'gloo' else (tensor.device if 'cuda' in str(tensor.device) else torch.device('cuda'))
    
    def distributed_model_rank_structure(self, subdomains, replicas_per_subdomain, stages_amount, gpus_per_sharded_layer, node_rank_dict):
        '''
        Outputs a list e.g., if we have 
            - 2 subdomains in data (Domain Decomposition in data)
            - 2 replicas per subdomain (data parallel within a subdomain)
            - 2 stages (pipeline)
            - 2 GPUs per stage (layer sharding - tensor parallelism) <- for improved performance it would be better to use this only in case multiple GPUs are available per node
        the output will be:
        [ 
            [ # Subdomain 0
                [ # Replica 0
                    [ # Stage 0
                        [0, 1], # Stage 0 is sharded across 2 GPUs
                    ], 
                    [ # Stage 1
                        [2, 3], # Stage 1 is sharded across 2 GPUs
                    ], 
                ], 
                [ # Replica 1
                    [ # Stage 0
                        [4, 5], # Stage 0 is sharded across 2 GPUs
                    ], 
                    [ # Stage 1
                        [6, 7], # Stage 1 is sharded across 2 GPUs
                    ], 
                ]
            ], 
            
            [ # Subdomain 1 (similar to Subdomain 0)
                ...
            ] 
        ] 
        
        '''
        gpus_per_rank = utils.check_gpus_per_rank()
        if gpus_per_sharded_layer > gpus_per_rank:
            raise ValueError("The number of GPUs per sharded layer cannot be greater than the number of GPUs per rank.")
        if gpus_per_rank % gpus_per_sharded_layer  == 0:
            # If the number of GPUs per rank is divisible by the number of GPUs per sharded layer, take all ranks from each node. This will not cause sharding across different nodes.
            rank_list = []
            for node in node_rank_dict:
                for i in node_rank_dict[node]:
                    rank_list.append(i)
        else:
            # Taking all the nodes may lead to sharding across different nodes. To avoid this, we take only the ranks from the first node.
            rank_list = []
            for node in node_rank_dict:
                for i in node_rank_dict[node][:gpus_per_sharded_layer]:
                    rank_list.append(i)
            if subdomains*replicas_per_subdomain*gpus_per_sharded_layer*stages_amount > len(rank_list):
                # If the amount of ranks is not large enough to run the model, take all the ranks from the first node.
                rank_list = []
                for node in node_rank_dict:
                    for i in node_rank_dict[node]:
                        rank_list.append(i)
                if self.rank == rank_list[0]:
                    print("Not enough GPUs per node to run the model. Taking all ranks from all nodes. NOTE that this may lead to sharding across different nodes, hence poor performance.")
        ranks = []; c = -1; layer_copies = {}; 
        subdomain_final_stages_main_rank = [[]]*subdomains
        all_final_stages_main_rank = []
        for sd in range(subdomains):
            subdomain_ranks = []
            for r in range(replicas_per_subdomain):
                replica_ranks = []
                for i in range(stages_amount):
                    stage = []
                    for j in range(gpus_per_sharded_layer):
                        c += 1
                        stage.append(rank_list[c])
                        if f'stage{i}_shard{j}' not in layer_copies:
                            layer_copies[f'stage{i}_shard{j}'] = [rank_list[c]]
                        else:
                            layer_copies[f'stage{i}_shard{j}'].append(rank_list[c])
                        if i == stages_amount - 1 and j == 0:
                            subdomain_final_stages_main_rank[sd].append(rank_list[c])
                        if i == stages_amount - 1 and j == 0:
                            all_final_stages_main_rank.append(rank_list[c])
                    replica_ranks.append(stage)
                subdomain_ranks.append(replica_ranks)
            ranks.append(subdomain_ranks)
        self.layer_copies = layer_copies
        self.all_model_ranks = ranks
        self.all_model_ranks_flat = utils.list_flattener(ranks)
        self.all_model_ranks_group = dist.new_group(ranks=self.all_model_ranks_flat, use_local_synchronization=True)
        
        # Store in each rank the correct layer_copies field - this will be needed to synchronize the parameters across the replicas
        self.last_layer_main_shard = layer_copies['stage'+str(stages_amount-1)+'_shard0']
        for layer in layer_copies:
            if 'stage'+str(stages_amount-1)+'_shard0' == layer: # last layers and main shard (0) are responsible for the computation of the loss
                self.last_layer_main_shard_group = dist.new_group(ranks=self.last_layer_main_shard, use_local_synchronization=True)
            if self.rank in layer_copies[layer]:
                self.all_layer_copies = layer_copies[layer]
                self.all_layer_copies_group = dist.new_group(ranks=self.all_layer_copies, use_local_synchronization=True)
                break     
            
        # create subdomain groups
        self.subdomain_ranks = self.subdomain_rank_structure(flatten=True)
        self.subdomain_ranks_group = dist.new_group(ranks=self.subdomain_rank_structure(flatten=True), use_local_synchronization=True)
        self.subdomain_final_stages_main_rank = [None]
        for sd in range(subdomains):
            if self.rank in subdomain_final_stages_main_rank[sd]:
                self.subdomain_final_stages_main_rank = subdomain_final_stages_main_rank[sd]
                self.subdomain_final_stages_main_rank_group = dist.new_group(ranks=self.subdomain_final_stages_main_rank, use_local_synchronization=True)
                break
        # create global group of final stages main rank
        self.all_final_stages_main_rank = all_final_stages_main_rank
        self.all_final_stages_main_rank_group = dist.new_group(ranks=self.all_final_stages_main_rank, use_local_synchronization=True)
        return ranks
    
    def from_rank_structure_to_layer_number(self):
        '''
        This function uses "self.all_model_ranks" and "self.rank" to return the layer number corresponding to the current rank.
        '''
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks: # replica_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]], e.g. stage_ranks = [0, 1]
                    if self.rank in stage_ranks: # sharded tensor
                        return replica_ranks.index(stage_ranks)
    
    def subdomain_rank_structure(self, flatten=False):
        '''
        This function returns the rank structure of the subdomains which contains the current self.rank.
        '''
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        if flatten:
                            return utils.list_flattener(subdomain_ranks)
                        else:
                            return subdomain_ranks
        return None
    
    def replica_rank_structure(self):
        '''
        This function returns the rank structure of the replicas which contains the current self.rank.
        '''
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        return replica_ranks
        return None
    
    def stage_rank_structure(self):
        '''
        This function returns the rank structure of the stages which contains the current self.rank.
        '''
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        return stage_ranks
        return None

