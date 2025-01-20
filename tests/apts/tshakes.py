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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from datasets.tinyshakespeare import *
from optimizers.apts import APTS
from optimizers.trust_region import TrustRegion
from pmw.model_handler import *
from pmw.parallelized_model import ParallelizedModel
from src.models.skgpt.model import GPT
from src.models.skgpt.trainer import Trainer
from src.utils import CfgNode as CN
from src.utils import closure, get_starting_info, set_seed, setup_logging

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


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        C.data_chunks_amount = 1
        C.percentage = 100.0
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        if dist.get_rank() == 0:
            print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def main(rank, world_size, args):
    # Initialize process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Or another free port
        rank=rank,
        world_size=world_size
    )

    # Print world size and rank
    print(f"Rank {rank}/{world_size-1}")

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_dict(args)
    config.merge_and_cleanup()
    if rank == 0: print(config)   
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
    print(model)
    print('asd')
    # # construct the trainer object
    # trainer = Trainer(config.trainer, model, train_dataset)

    # # iteration callback
    # def batch_end_callback(trainer):

    #     if trainer.iter_num % 10 == 0:
    #         print(
    #             f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    #     if trainer.iter_num % 500 == 0:
    #         # evaluate both the train and test score
    #         model.eval()
    #         with torch.no_grad():
    #             # sample from the model...
    #             context = "O God, O God!"
    #             x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
    #                 None, ...].to(trainer.device)
    #             y = model.generate(x, 500, temperature=1.0,
    #                                do_sample=True, top_k=10)[0]
    #             completion = ''.join([train_dataset.itos[int(i)] for i in y])
    #             print(completion)
    #         # save the latest model
    #         print("saving model")
    #         ckpt_path = os.path.join(config.system.work_dir, "model.pt")

    #         torch.save(model.state_dict(), ckpt_path)
    #         # revert model to training mode
    #         model.train()

    # trainer.set_callback('on_batch_end', batch_end_callback)

    # # run the optimization
    # trainer.run()

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test Script with Seed Argument")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=15)
    parser.add_argument("--num_subdomains", type=int, default=1)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_chunks_amount", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=0)
    # The following should be enabled only if we want to use a non-predefined model
    # parser.add_argument("--n_layer", type=int, default=6)
    # parser.add_argument("--n_head", type=int, default=6)
    # parser.add_argument("--n_embd", type=int, default=192)
    # 
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--num_workers", type=int, default=0)
    # Subdomain iterations
    parser.add_argument("--max_subdomain_iters", type=int, default=3)
    args = parser.parse_args()

    num_shards = 1
    num_subdomains = args.num_subdomains
    num_replicas_per_subdomain = args.num_replicas_per_subdomain
    num_stages = args.num_stages
    world_size = num_subdomains*num_replicas_per_subdomain*num_stages*num_shards
    print("World size: ", world_size)

    # Serialize args into a dictionary
    args_dict = vars(args)

    mp.spawn(
        main,
        args=(world_size, args_dict),
        nprocs=world_size,
        join=True
    )
