import math

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import autograd

from src.models.base_model import BaseModel
from src.pmw.base_pmw_model import BasePMWModel
from src.utils import is_function_module


# TODO: Implement this properly to actually perform sharding of the layers.
class ShardedLayer(BasePMWModel):
    def __init__(self, layer_dict, layer_ranks):
        super().__init__()
        self.layer_dict = layer_dict
        self.layer_ranks = layer_ranks

        # Decide if this is a function or a trainable nn.Module
        # isfunction(...) is from 'inspect' or 'torch.nn.functional'
        # Adjust as needed if you have a different test.
        obj = layer_dict['callable']['object']

        # If no sharding or if it's a pure function: just handle it on rank[0].
        if is_function_module(layer_dict) or len(layer_ranks) == 1:
            if self.rank == layer_ranks[0]:
                if is_function_module(layer_dict):
                    # Pure function (like relu)
                    self.layer = obj
                    print("This should never be printed in sharded layer...")
                else:
                    # Single-rank trainable module
                    try:
                        self.layer = obj(**layer_dict['callable']['settings']).to(self.tensor_device)
                    except:
                        print('asd')

                    # Optional initialization
                    if isinstance(self.layer, nn.Linear):
                        torch.nn.init.normal_(self.layer.weight, mean=0.0, std=0.02)
                        if self.layer.bias is not None:
                            torch.nn.init.zeros_(self.layer.bias)
                    elif isinstance(self.layer, nn.Embedding):
                        torch.nn.init.normal_(self.layer.weight, mean=0.0, std=0.02)
                    elif isinstance(self.layer, nn.LayerNorm):
                        torch.nn.init.zeros_(self.layer.bias)
                        torch.nn.init.ones_(self.layer.weight)

                    # Example of custom named-parameter initialization
                    if hasattr(self.layer, "named_parameters"):
                        for pn, p in self.layer.named_parameters():
                            if pn.endswith('c_proj.weight'):
                                torch.nn.init.normal_(
                                    p,
                                    mean=0.0,
                                    std=0.02 / math.sqrt(2 * BaseModel.n_layer)
                                )
        else:
            # Handle actual sharding here. 
            pass

        self.optim_groups = None  # Will be set later


    def configure_params(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        self.optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

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
    
    def as_model_dict(self):
        """
        Convert the model to a dictionary representation.
        """
        return self.layer_dict