import torch.nn as nn
from pmw.model import BaseModel

class TensorSharding(BaseModel):
    def __init__(self, module, rank_list, device_mesh, parallelize_plan):
        super(TensorSharding, self).__init__()
        self.module = module
        self.rank_list = rank_list
        self.device_mesh = device_mesh
        self.parallelize_plan = parallelize_plan
        

        for i, (layer, ranks) in enumerate(zip(module, rank_list)):
            if len(ranks) == 1:
                # module[i] = parallelize_module(layer, device_mesh, parallelize_plan)
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
            else:
                raise NotImplementedError("Tensor parallelism is not yet implemented.")

    def unshard(self, rank=None): # or to_local
        '''
        Send all shards to the rank specified.
        '''
        pass
    
    def send_shards(self, rank_list=[]):
        '''
        Shard and send tensor to the specified rank_list for paralellization.
        '''
        pass
    
    def multiply(self, a, b):
        '''
        Multiply the sharded tensors.
        '''
        pass
    
    def add(self, a, b): # TODO: Check if this overrides the + operator
        '''
        Add the sharded tensors.
        '''
        pass
 