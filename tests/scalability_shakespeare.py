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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src')) # Necessary on Daint

from dataloaders import GeneralizedDistributedDataLoader
from optimizers import APTS, TR
from models.nanoGPT import *
from llms_datasets.tiny_shakespeare import *
from pmw.parallelized_model import ParallelizedModel
from pmw.model_handler import *
import utils

##########
# TODO: REMOVE AFTER
import warnings
warnings.filterwarnings("ignore")
#########
bias = True

# num_subdomains = 1
# num_replicas_per_subdomain = 1
# num_stages = 2
num_shards = 1
TEST_ACCURACY = False


def main(rank=None, master_addr=None, master_port=None, world_size=None, **kwargs):
    # Scalability testing values & CSV file name relevant parameters
    num_subdomains = kwargs.get("num_subdomains", 2)
    num_replicas_per_subdomain = kwargs.get("num_replicas_per_subdomain", 2)
    num_stages = kwargs.get("num_stages", 1)
    trial = kwargs.get("trial", 0)
    learning_rate = kwargs.get("learning_rate", 0.1)
    # Other values
    num_epochs = kwargs.get("num_epochs", 40)
    seed = kwargs.get("seed", 2456456)  # Default seed if not provided
    batch_size = kwargs.get("batch_size", 64)
    data_chunks_amount = kwargs.get("data_chunks_amount", 2)
    block_size = kwargs.get("block_size", 256)
    vocab_size = kwargs.get("vocab_size", 0)
    n_layer = kwargs.get("n_layer", 1)
    n_head = kwargs.get("n_head", 2)
    n_embd = kwargs.get("n_embd", 384)
    dropout = kwargs.get("dropout", 0.0)

    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()
    # _________ Some parameters __________
    # NOTE: Setting a bach size lower than the dataset size will cause the two dataloader (sequential and parallel) to have different batches, hence different losses and accuracies
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # ____________________________________

    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank
    train_dataset_par, test_dataset_par, tokenizer = load_shakespeare(
        train_split=0.8, block_size=block_size)

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
        bias=bias
    )

    def criterion(outputs, targets):
        B, T, C = outputs.size()
        # loss = F.cross_entropy(outputs.view(B*T, C), targets.view(-1), ignore_index=-1)
        loss = F.cross_entropy(outputs.reshape(
            B * T, C), targets.reshape(-1), ignore_index=-1)
        return loss

    torch.manual_seed(seed)
    model_dict = get_model_dict(config)

    if num_stages == 1:
        # Go through every key in dictionary and set the field stage to 0
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0
    model_handler = ModelHandler(
        model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)
    # print the stages of the model
    print(f'(rank {rank}) Model stage_list: {model_handler.stage_list}\n')
    torch.manual_seed(seed)
    train_loader = GeneralizedDistributedDataLoader(
        model_handler=model_handler, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    torch.manual_seed(seed)
    test_loader = GeneralizedDistributedDataLoader(model_handler=model_handler, dataset=test_dataset_par, batch_size=len(
        test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)

    # Sharded first layer into [0,1] -> dataloader should upload batch to 0 only
    x_batch, _ = next(iter(train_loader))
    random_input = x_batch.to(device)

    par_model = ParallelizedModel(
        model_handler=model_handler, sample=random_input)
    # par_model.load_state_dict(os.path.join(
    #     'E:\\ML_APTS_saved_data', 'net_weights.pth'))
    # print(
    #     f'Norm of the model: {par_model.subdomain.weight_parallelized_model.parameters().norm()}')

    subdomain_optimizer = torch.optim.SGD
    glob_opt_params = {
        'lr': learning_rate,
        'max_lr': 1.0,
        'min_lr': 0.05,
        'nu': 0.5,
        'inc_factor': 2.0,
        'dec_factor': 0.5,
        'nu_1': 0.25,
        'nu_2': 0.75,
        'max_iter': 3,
        'norm_type': 2
    }
    par_optimizer = APTS(model=par_model, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults={'lr': learning_rate, 'momentum': 0.9},
                         global_optimizer=TR, global_optimizer_defaults=glob_opt_params, lr=learning_rate, max_subdomain_iter=2, dogleg=True, APTS_in_data_sync_strategy='average', step_strategy='mean')

    # To track epoch results
    epoch_results = []
    for epoch in range(0, num_epochs+1):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        loss_total_par = 0
        counter_par = 0

        # Measure epoch time
        start_time = time.time()

        # Parallel training loop
        for i, (x, y) in enumerate(train_loader):
            # Print progress in percentage rounded to two decimal places in the print using .2f
            if rank == 0:
                print(f"Progress: {100*(i/len(train_loader)):.2f}%")

            dist.barrier()
            # dist.barrier()
            x = x.to(device)
            y = y.to(device)

            def final_subdomain_closure(outputs, y=y):
                y_chunks = y.chunk(len(outputs))
                loss = []
                for i, o in enumerate(outputs):
                    loss.append(criterion(o, y_chunks[i]))
                return loss

            par_optimizer.zero_grad()
            counter_par += 1
            if epoch == 0:
                closuree = utils.closure(x, y, criterion, par_model, data_chunks_amount=data_chunks_amount, compute_grad=False)
                par_loss = closuree()
                loss_total_par += par_loss
            else:
                par_loss = par_optimizer.step(closure=utils.closure(
                    x, y, criterion=criterion, model=par_model, data_chunks_amount=data_chunks_amount, compute_grad=True),
                    final_subdomain_closure=final_subdomain_closure)

            loss_total_par += par_loss
            par_model.sync_params()

            if i > 9:
                break
            # print(f"(ACTUAL PARALLEL) {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in par_model.parameters()]))}")

        avg_loss = -1
        epoch_time = -1
        if rank == 0:
            avg_loss = loss_total_par/counter_par
            # Measure elapsed time in seconds
            epoch_time = time.time() - start_time
            print(
                f'Epoch {epoch}, Parallel avg loss: {avg_loss}, Time: {epoch_time}')

        # Parallel testing loop
        if TEST_ACCURACY:
            with torch.no_grad():  # TODO: Make this work also with NCCL
                correct = 0
                total = 0
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    closuree = utils.closure(
                        images, labels, criterion, par_model, compute_grad=False, zero_grad=True, return_output=True)
                    _, test_outputs = closuree()
                    if model_handler.is_last_stage():
                        test_outputs = torch.cat(test_outputs)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted ==
                                    labels.to(predicted.device)).sum().item()

                if model_handler.is_last_stage():
                    accuracy = 100 * correct / total
                    # here we dist all reduce the accuracy
                    accuracy = torch.tensor(accuracy).to(device)
                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM,
                                    group=model_handler.get_layers_copy_group(mode='global'))
                    accuracy /= len(model_handler.get_stage_ranks(
                        stage_name='last', mode='global'))
                    print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')

                    last_stage_ranks = model_handler.get_stage_ranks(
                        stage_name='last', mode='global')

                    # Send the accuracy from last_stage_ranks[0] to rank 0
                    if rank == last_stage_ranks[0]:
                        dist.send(tensor=accuracy, dst=0)
        else:
            accuracy = -1
            
        # Save epoch results
        if rank == 0:
            epoch_results.append(
                {'epoch': epoch, 'time': epoch_time, 'loss': avg_loss, 'accuracy': accuracy})

    # Make CSV file name
    csv_file_name = f"tshakespeare_t_{trial}_nsd_{num_subdomains}_nrs_{num_replicas_per_subdomain}_nst_{num_stages}_bs_{batch_size}.csv"

    # Save results to CSV
    if rank == 0:
        df_results = pd.DataFrame(epoch_results)

        df_results.to_csv(csv_file_name, index=False)
        print(f"Results saved to {csv_file_name}")

    # Make path file csv_file_name but with .pth instead of .csv
    model_dict_file_name = csv_file_name.replace('.csv', '.pth')
    par_model.save_state_dict(model_dict_file_name)


if __name__ == '__main__':
    if 1 == 1:
        parser = argparse.ArgumentParser(
            description="Test Script with Seed Argument")
        parser.add_argument("--trial", type=int, default=0)
        parser.add_argument("--num_epochs", type=int, default=40)
        parser.add_argument("--num_subdomains", type=int, default=1)
        parser.add_argument("--num_replicas_per_subdomain",
                            type=int, default=1)
        parser.add_argument("--num_stages", type=int, default=2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_chunks_amount", type=int, default=1)
        parser.add_argument("--block_size", type=int, default=128)
        parser.add_argument("--vocab_size", type=int, default=0)
        parser.add_argument("--n_layer", type=int, default=6)
        parser.add_argument("--n_head", type=int, default=2)
        parser.add_argument("--n_embd", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        args = parser.parse_args()
        main(trial=args.trial, num_epochs=args.num_epochs, num_subdomains=args.num_subdomains, num_replicas_per_subdomain=args.num_replicas_per_subdomain, num_stages=args.num_stages, seed=args.seed, batch_size=args.batch_size,
             data_chunks_amount=args.data_chunks_amount, block_size=args.block_size, vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, learning_rate=args.learning_rate)

    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        num_subdomains = 1
        num_replicas_per_subdomain = 4
        num_stages = 1 
        WORLD_SIZE = num_subdomains*num_replicas_per_subdomain*num_stages*num_shards
        mp.spawn(main, args=(MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)
