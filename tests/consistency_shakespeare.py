import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import datasets, transforms

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


block_size=256
vocab_size=0
n_layer=1
n_head=2
n_embd=384
dropout=0.2
bias=True
    
num_subdomains = 1
num_replicas_per_subdomain = 1
num_stages = 6
num_shards = 1
TEST_ACCURACY = False

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()
    # _________ Some parameters __________
    batch_size = 64 # NOTE: Setting a bach size lower than the dataset size will cause the two dataloader (sequential and parallel) to have different batches, hence different losses and accuracies
    data_chunks_amount = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 2456456
    torch.manual_seed(seed)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    learning_rage = 0.1
    # ____________________________________

    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    torch.manual_seed(0)
    train_dataset_par, test_dataset_par, tokenizer = load_shakespeare(train_split=0.8, block_size=block_size)
    
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
        loss = F.cross_entropy(outputs.reshape(B * T, C), targets.reshape(-1), ignore_index=-1)
        return loss
    
    torch.manual_seed(0)
    model_dict = get_model_dict(config)

    if num_stages == 1:
        # Go through every key in dictionary and set the field stage to 0
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0
    model_handler = ModelHandler(model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)
    # print the stages of the model
    print(f'(rank {rank}) Model stage_list: {model_handler.stage_list}\n')
    torch.manual_seed(0)
    train_loader = GeneralizedDistributedDataLoader(model_handler=model_handler, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    torch.manual_seed(0)
    test_loader = GeneralizedDistributedDataLoader(model_handler=model_handler, dataset=test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)

    # Sharded first layer into [0,1] -> dataloader should upload batch to 0 only
    torch.manual_seed(0)
    x_batch, _ = next(iter(train_loader))
    torch.manual_seed(0)
    random_input =  x_batch.to(device)

    torch.manual_seed(0)
    par_model = ParallelizedModel(model_handler=model_handler, sample=random_input)
    # Save the par_model state_dict
    dst_rank = 0
    par_dict = par_model.state_dict(dst_rank=dst_rank)
    if rank == dst_rank:
        global_model = GPTModelFromDict(model_dict)
        par_dict = {f'layers.{k}': v for k, v in par_dict.items()}
        global_model.load_state_dict(par_dict)
        global_model = global_model.to(device)
        sequential_train_loader = torch.utils.data.DataLoader(train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        sequential_test_loader = torch.utils.data.DataLoader(test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=learning_rage)

    # first check if the output of both models is the same using random_input
    if dist.get_world_size() == 1:
        torch.manual_seed(0)
        par_outputs = par_model(random_input)
        torch.manual_seed(0)
        global_outputs = global_model(random_input)
        print(torch.allclose(par_outputs[0], global_outputs, atol=1e-6))
        # print the norm of the difference
        print(f"Difference in relative norm: {torch.norm(par_outputs[0] - global_outputs)/torch.norm(global_outputs)}, norm of par_outputs: {torch.norm(par_outputs[0])}, norm of global_outputs: {torch.norm(global_outputs)}")

    par_optimizer = torch.optim.SGD(par_model.parameters(), lr=learning_rage)
    for epoch in range(40):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        loss_total_par = 0
        counter_par = 0
        # Parallel training loop
        for i, (x, y) in enumerate(train_loader):
            dist.barrier()
            # dist.barrier()
            x = x.to(device)
            y = y.to(device)

            # Gather parallel model norm
            par_optimizer.zero_grad()
            counter_par += 1
            torch.manual_seed(0)
            par_loss = par_optimizer.step(closure=utils.closure(
                x, y, criterion=criterion, model=par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)) 

            loss_total_par += par_loss
            par_model.sync_params()
            
            if i > 10:
                break
            # print(f"(ACTUAL PARALLEL) {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in par_model.parameters()]))}")

        if rank == 0:
            print(f'Epoch {epoch}, Parallel avg loss: {loss_total_par/counter_par}')
            
        # Parallel testing loop
        if TEST_ACCURACY:
            with torch.no_grad():  # TODO: Make this work also with NCCL
                correct = 0
                total = 0
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    torch.manual_seed(0)
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
                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM, group=model_handler.get_layers_copy_group(mode='global'))
                    accuracy /= len(model_handler.get_stage_ranks(stage_name='last', mode='global'))
                    print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')

        # -------------------------------------- Sequential training loop --------------------------------------
        if rank == dst_rank:
            loss = 0; counter = 0
            for i, (x, y) in enumerate(sequential_train_loader):
                counter += 1
                x = x.to(device)
                y = y.to(device)

                def closure():
                    global_optimizer.zero_grad()
                    torch.manual_seed(0)
                    global_outputs = global_model(x)
                    global_loss = criterion(global_outputs, y)
                    global_loss.backward()
                    return global_loss

                loss += global_optimizer.step(closure)
                if i > 10:
                    break
                
            print(f'Epoch {epoch}, Sequential loss: {loss/counter}')
            # params = [param for param in global_model.parameters()]
            # print(f"(ACTUAL SEQUENTIAL) param norm: {torch.norm(torch.cat([p.flatten() for p in global_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in global_model.parameters()]))}")

            # -------------------------------------- Sequential testing loop --------------------------------------
            if TEST_ACCURACY:
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in sequential_test_loader:
                        images, labels = images.to(device), labels.to(device)
                        torch.manual_seed(0)
                        outputs = global_model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    print(f'Epoch {epoch}, Sequential accuracy: {accuracy}')

if __name__ == '__main__':
    torch.manual_seed(0)
    if 1 == 2:
        main()
    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        WORLD_SIZE = num_subdomains*num_replicas_per_subdomain*num_stages*num_shards
        mp.spawn(main, args=(MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)
        
