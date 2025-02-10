import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets.cifar10 import CIFAR10Dataset
from src.models.cnn.simple_cnn import SimpleCNN
from src.models.cnn.trainer_pmw import Trainer
from src.pmw.model_handler import ModelHandler
from src.pmw.parallelized_model import ParallelizedModel
from src.utils import CfgNode as CN
from src.utils import (check_gpus_per_rank, detect_environment, dprint,
                       find_free_port, prepare_distributed_environment,
                       set_seed, setup_logging)


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = '../../saved_networks/cifar10/parsgd_pmw/'

    # data
    C.data = CIFAR10Dataset.get_default_config()
  
    # model
    C.model = SimpleCNN.get_default_config()

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

    # Datasets 
    train_dataset = CIFAR10Dataset(config.data)
    config.model.input_channels = train_dataset.get_input_channels()
    config.model.output_classes = train_dataset.get_output_classes()

    # Define the model
    model = SimpleCNN(config.model)
    dprint(model)

    # Model handler
    model_handler = ModelHandler(model.model_dict, config.model.num_subdomains, config.model.num_replicas_per_subdomain)
    config.trainer.model_handler = model_handler

    # Construct the parallel model (overwrite the model)
    sample_input = train_dataset.get_sample_input(config.trainer)
    model = ParallelizedModel(model_handler, sample=sample_input)

    # Define the trainer
    trainer = Trainer(config.trainer, model, train_dataset)

    # Define batch-end callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            dprint(f"Iteration {trainer.iter_num}, Loss: {trainer.loss:.4f}")
        if trainer.iter_num % 500 == 0:
            dprint("Saving model...")
            model.save_state_dict(os.path.join(config.system.work_dir, f"model_iter_{trainer.iter_num}.pt"))

    trainer.set_callback("on_batch_end", batch_end_callback)

    # Run training
    trainer.run()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Training with ParallelizedModel and Trainer")
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