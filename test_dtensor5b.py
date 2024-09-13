import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh, distribute_module
import os 
import torch.multiprocessing as mp


def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def main(rank, master_addr, master_port, world_size):
    # Initialize the distributed environment
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    # Create a device mesh
    mesh = init_device_mesh("cuda", (2,))

    # Define a tensor and shard it across devices
    x = torch.randn(1, 2)
    sharded_x = distribute_tensor(x, device_mesh=mesh, placements=[Shard(1)])

    # Define a simple model and distribute it across devices
    model = torch.nn.Linear(2, 2, bias=False)
    distributed_model = distribute_module(model, mesh)

    # Forward pass
    output = distributed_model(sharded_x)

    # Compute loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # print gradients of module
    for name, param in model.named_parameters():
        print(f"Rank {rank} - {name} - {param.grad._local_tensor}")
    
    print('asd')

    # Gradients are synchronized automatically


if __name__ == '__main__':  
    world_size = 2
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)
