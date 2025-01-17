import warnings
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

from pmw.dataloaders import GeneralizedDistributedDataLoader
from models.nanogpt.model import *
from llms_datasets.tiny_shakespeare import *
from pmw.parallelized_model import ParallelizedModel
from pmw.model_handler import *
import utils
from optimizers import APTS, TR

##########
# TODO: REMOVE AFTER
warnings.filterwarnings("ignore")
#########
bias = True

num_shards = 1
TEST_ACCURACY = False
hours = 24
total_hours_in_seconds = hours*60*60
save_threshold_in_seconds = total_hours_in_seconds/2


def main(rank=None, master_address=None, master_port=None, world_size=None, **kwargs):
    # General settings
    num_epochs = kwargs.get("num_epochs", 15)
    learning_rate = kwargs.get("learning_rate", 0.001)
    seed = kwargs.get("seed", 3407)  # Default seed if not provided
    batch_size = kwargs.get("batch_size", 64)
    data_chunks_amount = kwargs.get("data_chunks_amount", 1)
    # Percentage of the dataset to use
    percentage = kwargs.get("percentage", 100.0)
    num_workers = kwargs.get("num_workers", 0)
    # Scalability testing values
    trial = kwargs.get("trial", 0)
    num_subdomains = kwargs.get("num_subdomains", 2)
    num_replicas_per_subdomain = kwargs.get("num_replicas_per_subdomain", 1)
    num_stages = kwargs.get("num_stages", 13)
    num_sdi = kwargs.get("sdi", 2)  # Subdomain iterations
    # Transformer parameters
    block_size = kwargs.get("block_size", 256)
    n_layer = kwargs.get("n_layer", 1)
    n_head = kwargs.get("n_head", 2)
    n_embd = kwargs.get("n_embd", 384)
    dropout = kwargs.get("dropout", 0.0)

    # Directory to save results
    tinyshakespeare_dir = "../results/tinyshakespeare"

    # File names
    lr_str = str(learning_rate).replace('.', '_')  # Learning rate string
    perc_str = str(percentage).replace('.', '_')  # Percentage string
    base_file_name = f"ts_t_{trial}_nw_{num_workers}_bls_{block_size}_nl_{n_layer}_nh_{n_head}_ne_{n_embd}_ns_{num_subdomains}_nr_{num_replicas_per_subdomain}_st_{num_stages}_bs_{batch_size}_dc_{data_chunks_amount}_lr_{lr_str}_perc_{perc_str}_sdi_{num_sdi}_epochs_{num_epochs}_apts"
    # File to save epoch results
    epoch_file_name = os.path.join(
        tinyshakespeare_dir, f"{base_file_name}.csv")
    iter_file_name = epoch_file_name.replace(
        '.csv', '_iter.csv')  # File to save iteration results

    # Check if the model has already been trained
    starting_epoch, starting_num_iters, epoch_results, iter_results, starting_network = utils.get_starting_info(
        rank, base_file_name, epoch_file_name, num_epochs)

    # Initialize distributed environment
    utils.prepare_distributed_environment(
        rank, master_address, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()

    # NOTE: Setting a bach size lower than the dataset size will cause the two dataloader (sequential and parallel) to have different batches, hence different losses and accuracies
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Device rank
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    # Load dataset
    train_dataset_par, _, tokenizer = load_shakespeare(
        train_split=0.8, block_size=block_size, percentage=percentage)

    # Set configuration for the model
    config = GPTConfig(
        num_stages=num_stages,
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias
    )

    # Define the loss function
    def criterion(outputs, targets):
        B, T, C = outputs.size()
        # loss = F.cross_entropy(outputs.view(B*T, C), targets.view(-1), ignore_index=-1)
        loss = F.cross_entropy(outputs.reshape(
            B * T, C), targets.reshape(-1), ignore_index=-1)
        return loss

    # Reset seed to ensure the model is initialized with the same weights
    torch.manual_seed(seed)
    model_dict = get_model_dict(config)

    if num_stages == 1:
        # Go through every key in dictionary and set the field stage to 0
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0

    # Initialize model handler
    model_handler = ModelHandler(
        model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)

    if rank == 0:
        print(f"World size: {dist.get_world_size()}")
        # print(f'(rank {rank}) Model stage_list: {model_handler.stage_list}\n')

    # Initialize dataloaders
    torch.manual_seed(seed)
    train_loader = GeneralizedDistributedDataLoader(
        model_handler=model_handler, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    torch.manual_seed(seed)

    # Sharded first layer into [0,1] -> dataloader should upload batch to 0 only
    x_batch, _ = next(iter(train_loader))
    random_input = x_batch.to(device)

    # Initialize model
    par_model = ParallelizedModel(
        model_handler=model_handler, sample=random_input)

    # Load model if it exists
    if starting_network != "":
        par_model.load_state_dict(
            f"../saved_networks/{starting_network}")
        if rank == 0:
            print(f"Model loaded from {starting_network}")

    # Initialize optimizer
    glob_opt_params = {
        'lr': learning_rate,
        'max_lr': 1.0,
        'min_lr': 0.0001,
        'nu': 0.5,
        'inc_factor': 2.0,
        'dec_factor': 0.5,
        'nu_1': 0.25,
        'nu_2': 0.75,
        'max_iter': 3,
        'norm_type': 2
    }

    glob_opt = TR
    subdomain_optimizer = torch.optim.SGD
    par_optimizer = APTS(model=par_model, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults={'lr': learning_rate, 'momentum': 0.9},
                         global_optimizer=glob_opt, global_optimizer_defaults=glob_opt_params, lr=learning_rate, max_subdomain_iter=num_sdi, dogleg=True, APTS_in_data_sync_strategy='average', step_strategy='mean')

    # Compute number of iterations needed per epoch
    num_iters_per_epoch = len(train_loader)
    if rank == 0:
        print("Number of iterations per epoch: ", num_iters_per_epoch)
        print(
            f"Starting from epoch {starting_epoch} and iteration {starting_num_iters}")
    num_iters = starting_num_iters

    # Parallel training loop
    training_start_time = time.time()  # TODO: fix start time when resuming training
    for epoch in range(starting_epoch, num_epochs+1):
        dist.barrier()
        model_dict_file_name = epoch_file_name.replace(
            '.csv', f'_epoch_{epoch}.pth').replace(tinyshakespeare_dir, "../saved_networks")
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        epoch_start_time = time.time()
        loss_total_par = 0
        counter_par = 0
        for i, (x, y) in enumerate(train_loader):
            iter_start_time = time.time()
            dist.barrier()
            counter_par += 1
            x = x.to(device)
            y = y.to(device)

            # Build an estimate as to how much time it would take to finish the epoch
            if rank == 0 and i == 1:
                time_passed = (time.time() - training_start_time)
                time_per_iter = time_passed/(num_iters+1)
                time_left = time_per_iter*(len(train_loader))
                print(
                    f"Time left for epoch {epoch} is approximately {time_left:.2f} seconds")

            def final_subdomain_closure(outputs, y=y):
                y_chunks = y.chunk(len(outputs))
                loss = []
                for i, o in enumerate(outputs):
                    loss.append(criterion(o, y_chunks[i]))
                return loss

            if epoch == 0:
                closuree = utils.closure(
                    x, y, criterion, par_model, data_chunks_amount=data_chunks_amount, compute_grad=False)
                par_loss = closuree()
            else:
                par_optimizer.zero_grad()
                closuree = utils.closure(
                    x, y, criterion=criterion, model=par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)
                par_loss = par_optimizer.step(closure=closuree,
                                              final_subdomain_closure=final_subdomain_closure)
                num_iters += 1

            loss_total_par += par_loss
            par_model.sync_params()
            if rank == 0 and epoch > 0:
                iter_time = time.time() - iter_start_time
                iter_results.append(
                    {'iteration': num_iters, 'time': iter_time, 'loss': par_loss})

        # Compute average loss and epoch time
        avg_loss = -1
        epoch_time = -1
        if rank == 0:
            avg_loss = loss_total_par/counter_par
            epoch_time = time.time() - epoch_start_time  # Epoch time in seconds
            print(
                f'Epoch {epoch}, Parallel avg loss: {avg_loss}, Time: {epoch_time}')

        # Save epoch and iteration results
        if rank == 0:
            epoch_results.append(
                {'epoch': epoch, 'time': epoch_time, 'loss': avg_loss})
            df_results = pd.DataFrame(epoch_results)
            df_results.to_csv(epoch_file_name, index=False)
            print(f"Epoch results saved to {epoch_file_name}")

            if epoch == 0:
                iter_results.append(
                    {'iteration': 0, 'time': 0, 'loss': avg_loss})

            df_iter_results = pd.DataFrame(iter_results)
            df_iter_results.to_csv(iter_file_name, index=False)
            print(f"Iteration results saved to {iter_file_name}")

        # Save model state
        par_model.save_state_dict(model_dict_file_name)

        # Reset cache to avoid memory leaks
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test Script with Seed Argument")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_subdomains", type=int, default=2)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--data_chunks_amount", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=0)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--percentage", type=float, default=50.0)
    parser.add_argument("--num_workers", type=int, default=0)
    # Subdomain iterations
    parser.add_argument("--sdi", type=int, default=2)
    args = parser.parse_args()

    master_address = 'localhost'
    master_port = '12345'
    num_subdomains = 2
    num_replicas_per_subdomain = 1
    num_stages = 1
    WORLD_SIZE = num_subdomains*num_replicas_per_subdomain*num_stages*num_shards

    args = (
        master_address, master_port, world_size,
        args.trial, args.num_epochs, args.num_subdomains,
        args.num_replicas_per_subdomain, args.num_stages, args.seed,
        args.batch_size, args.data_chunks_amount, args.block_size,
        args.vocab_size, args.n_layer, args.n_head,
        args.n_embd, args.dropout, args.learning_rate,
        args.percentage, args.num_workers, args.sdi
    )

    mp.spawn(
        main,
        args=args,
        nprocs=WORLD_SIZE,
        join=True
    )
