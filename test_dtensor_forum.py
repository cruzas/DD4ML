import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh, distribute_module
import torch.nn.functional as F
import torch.autograd as autograd



class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input)) 
    

def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # Shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)
            

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Initialize device mesh (assuming 4 GPUs or devices)
    device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dim0", "dim1"))
    rowwise_placement=[Shard(1)]
    matrix = torch.eye(10, 10)
    sharded_matrix = distribute_tensor(matrix, device_mesh=device_mesh["dim1"], placements=rowwise_placement)

    first_stage_ranks = [0, 1]
    second_stage_ranks = [2, 3]
    if rank in [0,1]:
        input_tensor = torch.tensor([1,2], dtype=torch.float32).view(1,2).to('cuda')
        sharded_input_tensor = distribute_tensor(input_tensor, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        first_stage = Module()
        # Set default parameters to 1:
        for param in first_stage.parameters():
            param.data.fill_(1)
        distributed_first_stage = distribute_module(first_stage, device_mesh=device_mesh["dim1"], partition_fn=shard_params)
        
        # Process the input tensor
        output = distributed_first_stage(sharded_input_tensor)
        
        # Send output to ranks [2,3]
        output = output.to_local().cpu()
        dist.send(tensor=output, dst=second_stage_ranks[first_stage_ranks.index(rank)])
        
    if rank in [2,3]:
        second_stage_group = dist.new_group(ranks=[2,3], use_local_synchronization=True)
        second_stage = Module()
        # Set default parameters to 1:
        for param in second_stage.parameters():
            param.data.fill_(1)
        distributed_second_stage = distribute_module(second_stage, device_mesh=device_mesh["dim1"], partition_fn=shard_params)
        
        # Receive output from ranks [0,1]
        output = torch.ones(1, 2).cpu()
        sharded_output = distribute_tensor(output, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        temp = sharded_output.to_local().cpu()
        dist.recv(tensor=temp, src=first_stage_ranks[second_stage_ranks.index(rank)])
        sharded_output._local_tensor = temp.to('cuda')
        
        # Process the output tensor
        output = distributed_second_stage(sharded_output)
        
        # Make a tensor list with two elements
        output_all_pieces = [torch.zeros(1,1, dtype=torch.float32, device='cpu') for _ in range(2)]
        dist.all_gather(output_all_pieces, output.to_local().cpu(), group=second_stage_group)
        
        # Compute MSE loss
        output_all_pieces = torch.cat(output_all_pieces).to('cuda')
        loss = output_all_pieces.sum()

        # NOTE: We would like to do a backward pass using autograd here. This currently does not work. Why?
        grad_output = autograd.grad(loss, output, retain_graph=True)[0]
        for param in second_stage.parameters():
            grad = autograd.grad(output, param, grad_outputs=grad_output, retain_graph=True)[0]
            param.grad = grad
            
        # Then we would need a send and receive operation to send the gradients to the first stage and autograd again
        #...HELP!!!

if __name__ == '__main__':  
    world_size = 4
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)
