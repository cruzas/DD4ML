import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import time
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
    # shard fc1 and fc2
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
    time.sleep(rank)
    if rank in [0,1]:
        # Distribute module 0 across ranks [0,1]
        input_tensor = torch.tensor([1,2], dtype=torch.float32).view(1,2).to('cuda')
        sharded_input_tensor = distribute_tensor(input_tensor, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        module = Module()
        # set default parameters to 1:
        for param in module.parameters():
            param.data.fill_(1)
        distributed_first_stage = distribute_module(module, device_mesh=device_mesh["dim1"], partition_fn=shard_params)
        
        # process the input tensor
        output = distributed_first_stage(sharded_input_tensor)
        
        # send output to ranks [2,3]
        output = output.to_local().cpu()
        print(f"Rank {rank} sending output of shape {output.shape} to rank {second_stage_ranks[first_stage_ranks.index(rank)]}")
        dist.send(tensor=output, dst=second_stage_ranks[first_stage_ranks.index(rank)])
        print(f"Rank {rank} sent output {output}")
        
    if rank in [2,3]:
        # Distribute module 1 across ranks [2,3]
        second_stage_group = dist.new_group(ranks=[2,3], use_local_synchronization=True)
        module = Module()
        # set default parameters to 1:
        for param in module.parameters():
            param.data.fill_(1)
        distributed_second_stage = distribute_module(module, device_mesh=device_mesh["dim1"], partition_fn=shard_params)
        
        # receive output from ranks [0,1]
        output = torch.ones(1, 2).cpu()
        sharded_output = distribute_tensor(output, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        temp = sharded_output.to_local().cpu()
        print(f"Rank {rank} receiving output of shape {temp.shape} from rank {first_stage_ranks[second_stage_ranks.index(rank)]}")
        dist.recv(tensor=temp, src=first_stage_ranks[second_stage_ranks.index(rank)])
        print(f"Rank {rank} received output from rank {first_stage_ranks[second_stage_ranks.index(rank)]}")
        sharded_output._local_tensor = temp.to('cuda')
        
        # process the output tensor
        output = distributed_second_stage(sharded_output)
        print(f"Rank {rank} output:\n", output)
        
        # Make a tensor list with two elements
        output_all_pieces = [torch.zeros(1,1, dtype=torch.float32, device='cpu') for _ in range(2)]
        dist.all_gather(output_all_pieces, output.to_local().cpu(), group=second_stage_group)
        
        # unshard the output tensor
        print(f"Rank {rank} output_all_pieces:\n", output_all_pieces)
        
        # Compute MSE loss
        output_all_pieces = torch.cat(output_all_pieces).to('cuda')
        target = torch.zeros_like(output_all_pieces).to('cuda')  # Assume the target is zeros
        loss = output_all_pieces.sum()
        print(f"Rank {rank} loss: {loss.item()}")

        # Backward pass manually through autograd
        loss.backward()
    
    # here compute the loss through MSE and then do a backward pass

if __name__ == '__main__':  
    world_size = 4
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)
