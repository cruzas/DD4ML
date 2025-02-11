import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import detect_environment, prepare_distributed_environment

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch version: {torch.__version__}")

# Try to import wandb
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    print("WandB is not available. Logging to stdout.")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main(rank, master_addr, master_port, world_size, args=None):
    # Initialize the distributed environment
    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Global rank: {rank}, Local rank: {local_rank}")

    # Initialize wandb if available on rank 0
    if wandb_available and rank == 0:
        wandb.init(project="my_project", config={
            "epochs": 3,
            "batch_size": 250,
            "learning_rate": 0.01
        })

    model = SimpleCNN().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=250, sampler=sampler)

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    ddp_model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        for inputs, targets in dataloader:
            inputs = inputs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(ddp_model(inputs), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Reduce epoch loss and batch count across all processes
        epoch_loss_tensor = torch.tensor(epoch_loss, device=local_rank)
        num_batches_tensor = torch.tensor(num_batches, device=local_rank)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        average_loss = epoch_loss_tensor.item() / num_batches_tensor.item()

        if rank == 0:
            print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}")
            if wandb_available:
                wandb.log({"epoch": epoch+1, "average_loss": average_loss})

    dist.destroy_process_group()

    if wandb_available and rank == 0:
        wandb.finish()

if __name__ == '__main__':
    environment = detect_environment()
    rank = None
    master_addr = None 
    master_port = None
    world_size = None
    print("Code being executed on a cluster...")
    main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size)