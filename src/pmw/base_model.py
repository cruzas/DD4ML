from abc import abstractmethod
import torch.nn as nn
import torch.distributed as dist
import torch


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.backend = dist.get_backend()
        self.tensor_device = 'cuda'
        self.default_device = 'cpu' # Default device to store scalars
        
    def backend_device(self, tensor=torch.tensor([0])):
        return 'cpu' if dist.get_backend() == 'gloo' else (tensor.device if 'cuda' in tensor.device else 'cuda')
