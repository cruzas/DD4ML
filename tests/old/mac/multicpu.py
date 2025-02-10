import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP


def train(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Or another free port
        rank=rank,
        world_size=world_size
    )

    # Create a simple model
    model = nn.Linear(10, 5)
    ddp_model = DDP(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    # Dummy data
    data = torch.randn(16, 10)
    target = torch.randn(16, 5)

    # Basic training loop
    for epoch in range(2):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Only let rank 0 print
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    # Adjust world_size for the number of CPU cores you want to use
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
