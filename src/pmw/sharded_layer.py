import math
from inspect import isfunction

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import autograd

from src.pmw.base_model import BaseModel


# TODO: Implement this properly to actually perform sharding of the layers.
class ShardedLayer(BaseModel):
    def __init__(self, layer_dict, layer_ranks):
        super().__init__()

        self.layer_dict = layer_dict
        self.layer_ranks = layer_ranks
        if isfunction(layer_dict['callable']['object']) or len(layer_ranks) == 1:
            # This has no sharding so it will run on the main shard rank only
            if self.rank == layer_ranks[0]:
                if isfunction(layer_dict['callable']['object']):
                    self.layer = layer_dict['callable']['object']
                else:
                    self.layer = layer_dict['callable']['object'](**layer_dict['callable']['settings']).to(self.tensor_device)
                
                if isinstance(self.layer, nn.Linear):
                    torch.nn.init.normal_(self.layer.weight, mean=0.0, std=0.02)
                    if self.layer.bias is not None:
                            torch.nn.init.zeros_(self.layer.bias)
                elif isinstance(self.layer, nn.Embedding):
                    torch.nn.init.normal_(self.layer.weight, mean=0.0, std=0.02)
                elif isinstance(self.layer, nn.LayerNorm):
                    torch.nn.init.zeros_(self.layer.bias)
                    torch.nn.init.ones_(self.layer.weight)

                for pn, p in self.layer.named_parameters():
                    if pn.endswith('c_proj.weight'):
                        torch.nn.init.normal_(
                            p, mean=0.0, std=0.02/math.sqrt(2 * BaseModel.n_layer))
        else:
            # This has sharding...TODO
            pass

    def forward(self, x):
        """
        Forward pass of the sharded layer.
        Args:
            x: Input to the layer.
        Returns:
            torch.Tensor: Output of the layer.
        """
        # TODO/IMPORTANT: Implement the forward pass for sharded. Also adapt it to the case where we have mixed stages with functions (only one rank) and layers (multiple ranks)
        out = self.layer(x)
        return out 

    
    def backward(self, output, grad_output, num_chunks):
        if not self.layer_is_sharded:
            for param in self.parameters():
                grad = autograd.grad(output, param, grad_outputs=grad_output, retain_graph=True)[0] / num_chunks
                param.grad = grad if param.grad is None else param.grad + grad
        else:
            raise NotImplementedError("Sharded layer backward pass is not implemented yet.")


    def unshard(self, gpu_id=0): 
        """
        Send all shards to a specific GPU in the current rank.
        """
        pass
    
    def send_shards(self, dst):
        """
        Shard and send tensor to the specified rank.
        """
        # NOTE: Remember to implement two strategies, NCCL/GLOO where in case of GLOO everything 
        # is sent to CPU first (find a way to know where to send the shards upon arrival to the destination rank)
        # USE THIS: https://pytorch.org/docs/stable/distributed.html#torch.distributed.send_object_list
        pass
    
    def receive_shards(self, src):
        """
        Receive shards from the specified rank.
        """
        # USE THIS: https://pytorch.org/docs/stable/distributed.html#torch.distributed.recv_object_list
        pass