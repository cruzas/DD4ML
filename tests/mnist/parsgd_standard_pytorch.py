import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.utils import (detect_environment, dprint, get_rawdata_dir,
                       prepare_distributed_environment)


# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function for training a single epoch
def train_epoch(rank, model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress within the epoch
        total_batches = len(train_loader)
        progress = int(100 * (batch_idx + 1) / total_batches)
        dprint(f"Epoch train [{epoch+1}/{num_epochs}] {progress}%\r")

    return running_loss / len(train_loader)

# Function for testing the model
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        progress = int(100 * (batch_idx + 1) / len(test_loader))
        dprint(f"Test progress: {progress}%\r")

    accuracy = 100.0 * correct / total
    return accuracy

# Main function for data-parallel training
def main(rank, master_addr, master_port, world_size, args):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the process group
    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())

    # Define MNIST dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    rawdata_dir = get_rawdata_dir()
    train_dataset = torchvision.datasets.MNIST(root=rawdata_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=rawdata_dir, train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], sampler=train_sampler, num_workers=args["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])

    # Define the model (Simple CNN)
    model = SimpleCNN().to(device)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(args["num_epochs"]):
        train_loader.sampler.set_epoch(epoch)
        running_loss = train_epoch(rank, model, train_loader, criterion, optimizer, device, epoch, args["num_epochs"])

        dprint(f"\nEpoch [{epoch+1}/{args['num_epochs']}], Loss: {running_loss:.4f}")

        # Test the model after each epoch
        test_accuracy = test(model, test_loader, device)
        dprint(f"Test Accuracy: {test_accuracy:.2f}%")

    dprint("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST Training Script with Simple CNN")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--num_workers", type=int, default=1)
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
