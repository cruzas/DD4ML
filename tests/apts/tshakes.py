import argparse
import json
import os
import random
import sys
import time
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd  # IMPORTANT: this should come after numpy!
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

from datasets.char_dataset import *
from src.models.gpt.skgpt.model import GPT
from src.models.gpt.skgpt.trainer import Trainer
from src.optimizers.apts import APTS
from src.optimizers.trust_region import TrustRegion
from src.pmw.base_model import BaseModel
from src.pmw.model_handler import ModelHandler
from src.pmw.parallelized_model import ParallelizedModel
from src.utils import (closure, detect_environment, dprint, get_starting_info,
                       prepare_distributed_environment, set_seed,
                       setup_logging)

# Some settings
bias = True

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = '../../saved_networks/tshakes/'

    # data
    C.data = CharDataset.get_default_config()
  
    # model
    C.model = GPT.get_default_config()

    # trainer
    C.trainer = Trainer.get_default_config()
    return C

def get_sample_input(train_dataset, config):
    dummy_train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    x_batch, _ = next(iter(dummy_train_loader))
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return x_batch.to(device)

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
    config.merge_and_cleanup()
    dprint(config)   
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    # don't worry we won't run out of file handles
    text = open('../../input.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    BaseModel.n_layer = config.model.n_layer # Bit of a hack for now
    dprint(model)

    # construct the model handler
    model_handler = ModelHandler(model.model_dict, config.model.num_subdomains, config.model.num_replicas_per_subdomain)
    config.trainer.model_handler = model_handler

    # construct the parallel model (overwrite the model)
    random_input = get_sample_input(train_dataset, config.trainer)
    model = ParallelizedModel(model_handler, sample=random_input)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    
    # iteration callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            dprint(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss:.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                # context = "O God, O God!" # TODO: check why this doesn't work with our model
                # x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                x = random_input
                if x.size(0) > 1:
                    x = x[0:1]
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                dprint(completion)
            # save the latest model
            dprint("saving model")
            model_filename = f"model_{config.model.model_type}_nsd_{config.model.num_subdomains}_nr_{config.model.num_replicas_per_subdomain}_nst_{config.model.num_stages}_s_{config.system.seed}_t_{config.system.trial}_i_{trainer.iter_num}.pt"
            ckpt_path = os.path.join(config.system.work_dir, model_filename)

            model.save_state_dict(ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test Script with Seed Argument")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--num_subdomains", type=int, default=1)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_chunks_amount", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=0)
    # The following should be enabled only if we want to use a non-predefined model
    # parser.add_argument("--n_layer", type=int, default=6)
    # parser.add_argument("--n_head", type=int, default=6)
    # parser.add_argument("--n_embd", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_subdomain_iters", type=int, default=3)
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
        world_size = args.num_subdomains * args.num_replicas_per_subdomain * args.num_stages * args.num_shards
        mp.spawn(
            main,
            args=(master_addr, master_port, world_size, args_dict),
            nprocs=world_size,
            join=True
        )
    else:
        print("Code being executed on a cluster...")
        main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args_dict)
