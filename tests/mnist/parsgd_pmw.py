import argparse
import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import wandb
from src.datasets.mnist import MNISTDataset
from src.models.cnn.mnist import CNNMNIST
from src.models.cnn.mnist.trainer_pmw import Trainer
from src.pmw.model_handler import ModelHandler
from src.pmw.parallelized_model import ParallelizedModel
from src.utils import CfgNode as CN
from src.utils import (check_gpus_per_rank, detect_environment, dprint,
                       find_free_port, prepare_distributed_environment,
                       set_seed, setup_logging)

filename = os.path.basename(__file__).split(".")[0]

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f'../../saved_networks/mnist/{filename}/'

    # data
    C.data = MNISTDataset.get_default_config()
  
    # model
    C.model = CNNMNIST.get_default_config()

    # trainer
    C.trainer = Trainer.get_default_config()
    return C

# Main function
def main(rank=None, master_addr=None, master_port=None, world_size=None, args=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())
    print(f"Rank {rank}/{world_size-1}")

    if torch.cuda.is_available() and args['num_shards'] > 1:
        # Number of GPUs should be the same on every rank
        check_gpus_per_rank()

    config = get_config()
    config.merge_from_dict(args)
    config.merge_and_cleanup(keys_to_look=["system", "model", "trainer"])
    dprint(config)   
    setup_logging(config)
    set_seed(config.system.seed)

    # Initialize wandb
    if rank == 0:  # Only initialize wandb for the main process
        wandb.init(project=f"mnist-training-{filename}", config=args, name=f"trial_{args['trial']}")

    # Datasets 
    train_dataset = MNISTDataset(config.data)
    config.model.input_channels = train_dataset.get_input_channels()
    config.model.output_classes = train_dataset.get_output_classes()
    
    test_dataset_config = copy.deepcopy(config.data)
    test_dataset_config.train = False
    test_dataset = MNISTDataset(test_dataset_config)

    # Define the model
    model = CNNMNIST(config.model)
    dprint(model)

    # Model handler
    model_handler = ModelHandler(model.model_dict, config.model.num_subdomains, config.model.num_replicas_per_subdomain)
    config.trainer.model_handler = model_handler

    # Construct the parallel model (overwrite the model)
    sample_input = train_dataset.get_sample_input(config.trainer)
    model = ParallelizedModel(model_handler, sample=sample_input)

    # Define optimizer
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)

    # Define epoch-end callback
    def epoch_end_callback(trainer):
        dprint(f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt:.2f}s")
        if rank == 0:  # Log only from the main process
            wandb.log({
                "epoch": trainer.epoch_num,
                "loss": trainer.loss,
                "accuracy": trainer.accuracy,
                "epoch_time": trainer.epoch_dt
            })
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

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Training with ParallelizedModel and Trainer")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    # for pmw
    parser.add_argument("--num_subdomains", type=int, default=1)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    # Serialize args into a dictionary for passing them to main
    args_dict = vars(args)

    # Environment we are in
    environment = detect_environment()
    rank = None
    master_addr = None
    master_port = None
    world_size = None

    if environment == "local":
        print("Code being executed locally...")
        master_addr = "localhost"
        master_port = find_free_port()
        world_size = args.num_subdomains * args.num_replicas_per_subdomain * args.num_stages * args.num_shards
        mp.spawn(main, args=(master_addr, master_port, world_size, args_dict), nprocs=world_size, join=True)
    else:
        print("Code being executed on a cluster...")
        main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args_dict)
