import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.utils import detect_environment, prepare_distributed_environment


# Function for training a single epoch
def train_epoch(rank, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(rank), targets.to(rank)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Function for testing the model
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

# Main function for data-parallel training
def main(rank, master_addr, master_port, world_size, args):
    torch.manual_seed(0)

    # Set up the process group
    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())


    # Define CIFAR-10 dataset and data loaders
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Define the model (ResNet-18)
    model = resnet18(num_classes=10).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        running_loss = train_epoch(rank, model, train_loader, criterion, optimizer)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

            # Test the model after each epoch
            test_accuracy = test(model, test_loader, rank)
            print(f"Test Accuracy: {test_accuracy:.2f}%")

    if rank == 0:
        print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Script with Seed Argument")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Serialize args into a dictionary for passing them to main
    args_dict = vars(args)

    # Environment we are in
    environment = detect_environment()

    # For distributed environment initialization
    rank = None
    master_addr = None 
    master_port = None
    world_size = None
    if environment == "local":
        print("Code being executed locally...")
        master_addr = "localhost"
        master_port = "29501"
        world_size = 2
        mp.spawn(
            main,
            args=(master_addr, master_port, world_size, args_dict),
            nprocs=world_size,
            join=True
        )
    else:
        print("Code being executed on a cluster...")
        main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args_dict)

