import torch.nn as nn
from pmw.model import BaseModel
from torch import autograd
import torch.autograd as autograd

# TODO: Implement this properly to actually perform sharding of the layers.
class ShardedLayer(BaseModel):
    def __init__(self, layer, is_sharded:bool):
        super(ShardedLayer, self).__init__()
        self.layer = layer
        # TODO: Handle this outside this class.. use nn.Sequential instead.
        self.is_sharded = is_sharded

        # TODO: Check the type of a layer and shard it, if sharding is not possible set a variable 
        # to False in order to know that the layer is not sharded so "unshard" can be called on the dst rank.
        if self.is_sharded:
            if isinstance(layer, nn.Linear):
                self.layer_is_sharded = False # This should be a True (after proper implementation)
                pass
            else:
                self.layer_is_sharded = False
        else:
            self.layer_is_sharded = False

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
    
    def forward(self, x):
        """
        Forward pass of the sharded layer.
        Args:
            x: Input to the layer.
        Returns:
            torch.Tensor: Output of the layer.
        """
        if not self.layer_is_sharded:
            # TODO: (ExaTrain 2) Store in the current class the input/output of the layer to be able make a subdomain out of this sharded layer.
            return self.layer(x)
        else:
            raise NotImplementedError("Sharded layer forward pass is not implemented yet.") 

    def backward(self, output, grad_output, num_chunks):
        if not self.layer_is_sharded:
            for param in self.parameters():
                grad = autograd.grad(output, param, grad_outputs=grad_output, retain_graph=True)[0] / num_chunks
                param.grad = grad if param.grad is None else param.grad + grad
        else:
            raise NotImplementedError("Sharded layer backward pass is not implemented yet.")