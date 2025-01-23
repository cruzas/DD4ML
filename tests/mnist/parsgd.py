import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models.cnn.trainer import Trainer
from src.utils import CfgNode as CN
from src.utils import (detect_environment, dprint, find_free_port,
                       get_rawdata_dir, prepare_distributed_environment,
                       set_seed, setup_logging)


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = '../../saved_networks/chargpt/'

    # data
    # C.data = MNISTDataset.get_default_config()

    # model
    C.model = CN()

    # trainer
    C.trainer = Trainer.get_default_config()
    # the model we're using is so small that we can go a bit faster
    C.trainer.learning_rate = 1e-3
    C.trainer.momentum = 0.9

    return C

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


# Main function for data-parallel training
def main(rank, master_addr, master_port, world_size, args):
    # Initialize distributed environment
    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())
    print(f"Rank {rank}/{world_size-1}")

    if torch.cuda.is_available() and args['num_shards'] > 1:
        # Number of GPUs should be the same on every rank
        check_gpus_per_rank()

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_dict(args)
    config.merge_and_cleanup(keys_to_look=['system', 'model', 'trainer'])
    dprint(config)   
    setup_logging(config)
    set_seed(config.system.seed)

    # Define MNIST dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    rawdata_dir = get_rawdata_dir()
    train_dataset = torchvision.datasets.MNIST(root=rawdata_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=rawdata_dir, train=False, download=True, transform=transform)

    # Define the model (Simple CNN)
    model = SimpleCNN()
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Define loss function and optimizer
    config.trainer.rank = rank
    config.trainer.world_size = world_size
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)

    # Define epoch-end callback
    def epoch_end_callback(trainer):
        dprint(f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt:.2f}s")
        if trainer.epoch_num % 5 == 0:
            dprint("Saving model...")
            model_path = os.path.join(config.system.work_dir, f"model_epoch_{trainer.epoch_num}.pt")
            torch.save(model.module.state_dict(), model_path)

    # Define batch-end callback
    def batch_end_callback(trainer):
        dprint(f"Epoch [{trainer.epoch_num}/{trainer.config.num_epochs}] {trainer.epoch_progress}%\r")

    trainer.set_callback("on_epoch_end", epoch_end_callback)
    trainer.set_callback("on_batch_end", batch_end_callback)

    # Run training
    trainer.run()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST Training Script with Simple CNN")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
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
        master_addr = "localhost"
        master_port = find_free_port()
        world_size = 2
        print(f"Code being executed locally with {world_size} process(es)...")
        mp.spawn(
            main,
            args=(master_addr, master_port, world_size, args_dict),
            nprocs=world_size,
            join=True
        )
    else:
        print("Code being executed on a cluster...")
        main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args_dict)
