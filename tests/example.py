import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

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

def train():
    # Environment variables set by torchrun
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Initialize process group using the env:// method.
    dist.init_process_group(backend='nccl', init_method='env://')

    # Print global and local rank
    print(f'Global rank: {rank}, Local rank: {local_rank}')

    model = SimpleCNN().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=sampler)

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    ddp_model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for inputs, targets in dataloader:
            inputs = inputs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(ddp_model(inputs), targets)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == '__main__':
    train()