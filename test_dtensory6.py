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

    # Define a tensor with known values and shard it across devices
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)  # Input tensor with known values
    sharded_x = distribute_tensor(x, device_mesh=mesh, placements=[Shard(1)])

    # Define a simple Linear model (without bias) and distribute it across devices
    model = torch.nn.Linear(2, 2, bias=False)  # Linear layer with no bias
    distributed_model = distribute_module(model, mesh)

    # Manually set weights to known values (e.g., W = [[1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    # Forward pass
    output = distributed_model(sharded_x)

    # Compute loss as the sum of outputs
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Print the gradients of the model weights
    for name, param in model.named_parameters():
        print(f"Rank {rank} - {name} - Gradient: {param.grad}")

    # Expected gradient: for W, it should be equal to the input tensor x.
    dist.barrier()


if __name__ == '__main__':
    world_size = 2
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)
