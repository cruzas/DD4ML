import os
import sys
import argparse
import time
import pandas as pd
import torch
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import datasets, transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloaders import GeneralizedDistributedDataLoader
from models.nanoGPT import *
from llms_datasets.tiny_shakespeare import *
from pmw.parallelized_model import ParallelizedModel
from pmw.model_handler import *
import utils

import warnings
warnings.filterwarnings("ignore")

def main(rank=None, master_addr=None, master_port=None, world_size=None, **kwargs):
    # Parameters
    num_subdomains = kwargs.get("num_subdomains", 1)
    num_replicas_per_subdomain = kwargs.get("num_replicas_per_subdomain", 1)
    num_stages = kwargs.get("num_stages", 1)
    seed = kwargs.get("seed", 0)
    batch_size = kwargs.get("batch_size", 64)
    block_size = kwargs.get("block_size", 128)
    n_layer = kwargs.get("n_layer", 6)
    n_head = kwargs.get("n_head", 2)
    n_embd = kwargs.get("n_embd", 256)
    dropout = kwargs.get("dropout", 0.0)
    data_chunks_amount = kwargs.get("data_chunks_amount", 1)
    load_path = kwargs.get("load_path", "model_dict.pth")  # Path to load model dict

    utils.prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank
    train_dataset_par, test_dataset_par, tokenizer = load_shakespeare(train_split=0.8, block_size=block_size)

    if rank == 0:
        print(f"World size: {dist.get_world_size()}")

    config = GPTConfig(
        num_stages=num_stages,
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True
    )

    def criterion(outputs, targets):
        B, T, C = outputs.size()
        loss = F.cross_entropy(outputs.reshape(B * T, C), targets.reshape(-1), ignore_index=-1)
        return loss

    torch.manual_seed(seed)
    model_dict = get_model_dict(config)

    # If only one stage, set all to stage 0
    if num_stages == 1:
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0

    model_handler = ModelHandler(model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)
    print(f'(rank {rank}) Model stage_list: {model_handler.stage_list}\n')

    # No training here, only testing
    test_loader = GeneralizedDistributedDataLoader(
        model_handler=model_handler,
        dataset=test_dataset_par,
        batch_size=len(test_dataset_par),
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Create a dummy input for model initialization
    dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, block_size), dtype=torch.long).to(device)
    par_model = ParallelizedModel(model_handler=model_handler, sample=dummy_input)

    # Load model state dict
    par_model.load_state_dict(load_path)

    # Testing accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            # Print progress 
            if rank == 0:
                print(f"Progress: {total}/{len(test_dataset_par)}", end='\r')
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            closure_fn = utils.closure(
                images, labels, criterion, par_model, compute_grad=False, zero_grad=True, return_output=True
            )
            _, test_outputs = closure_fn()
            if model_handler.is_last_stage():
                test_outputs = torch.cat(test_outputs)
                _, predicted = torch.max(test_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(predicted.device)).sum().item()

        if model_handler.is_last_stage():
            accuracy = 100 * correct / total
            accuracy_tensor = torch.tensor(accuracy).to(device)
            dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM,
                            group=model_handler.get_layers_copy_group(mode='global'))
            accuracy_tensor /= len(model_handler.get_stage_ranks(stage_name='last', mode='global'))
            if rank == 0:
                print(f"Test Accuracy: {accuracy_tensor.item()}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Only Script")
    parser.add_argument("--num_subdomains", type=int, default=1)
    parser.add_argument("--num_replicas_per_subdomain", type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_chunks_amount", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--load_path", type=str, default="model_dict.pth")
    args = parser.parse_args()
    main(num_subdomains=args.num_subdomains, num_replicas_per_subdomain=args.num_replicas_per_subdomain,
         num_stages=args.num_stages, seed=args.seed, batch_size=args.batch_size,
         data_chunks_amount=args.data_chunks_amount, block_size=args.block_size,
         n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
         dropout=args.dropout, load_path=args.load_path)
