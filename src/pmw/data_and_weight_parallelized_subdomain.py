import torch
import torch.distributed as dist
import utils
from pmw.base_model import BaseModel
from pmw.weight_parallelized_model import WeightParallelizedModel

class DataAndWeightParallelizedSubdomain(BaseModel):
    def __init__(self, model_handler, sample):
        super().__init__()
        '''
        This function defines a subdomain in data. The subdomain has subdomains in weights and can parallelize forward and backward passes in data.
        
        rank_list = list of lists containing the ranks of the model replicas that are responsible for each layer, e.g. [[0, 1], [2, 3], [4, 5]] (3 replicas, 2 layers each replica)
        '''
        self.model_handler = model_handler
        self.num_replicas_per_subdomain = self.model_handler.num_replicas_per_subdomain        
        self.weight_parallelized_model = WeightParallelizedModel(model_handler=model_handler, sample=sample)
        # Synchronize the parameters across all replicas
        self.sync_params() 
        
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.weight_parallelized_model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses, sync=True):
        self.weight_parallelized_model.backward(losses=losses)
        if sync:
            self.sync_grads()
    
    def sync_params(self, method='average'):
        # TODO/NOTE: Use the sharding class 
        if self.num_replicas_per_subdomain > 1:
            for param in self.weight_parallelized_model.subdomain.parameters():
                dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                if method == 'average':
                    param.data /= self.num_replicas_per_subdomain
                elif method == 'sum':
                    pass # nothing to do since we already summed the parameters through all_reduce
                else:
                    raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        if self.num_replicas_per_subdomain > 1:
            for param in self.weight_parallelized_model.subdomain.parameters():
                dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.grad /= self.num_replicas_per_subdomain