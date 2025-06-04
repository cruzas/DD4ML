import inspect
import os
from types import FunctionType

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def get_device(device, backend="cuda"):
    if device is not None:
        return device
    else:
        return (
            f"cuda:{torch.cuda.current_device()}"
            if backend != "gloo"
            else "cpu"
        )

def cross_entropy_transformers(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

# Helper to detect function-only modules (no learned parameters).
def is_function_module(info):
    """
    Return True if 'info' is for a 'function-like' module (e.g. relu),
    False if it is a trainable nn.Module.
    """
    obj = info['callable']['object']
    
    # If 'obj' is a class inheriting from nn.Module, it has parameters.
    if inspect.isclass(obj) and issubclass(obj, nn.Module):
        return False
    # If it's a Python function or a custom label (like 'method_view'),
    # treat it as function-only (param-free).
    if isinstance(obj, FunctionType) or isinstance(obj, str):
        return True
    return False


def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False, data_chunks_amount=1, grad_norm_clip=None, outputs_only=False):
    """
    NOTE: Losses from different chunks are averaged.
    """
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')

    has_model_handler = hasattr(model, 'model_handler')

    if has_model_handler and model.model_handler.is_last_stage() and targets is not None and not outputs_only:
        targets = targets.chunk(data_chunks_amount)

    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount, sync_loss='global', grad_norm_clip=grad_norm_clip, outputs_only=outputs_only):
        '''
        sync_loss: 'global' or 'local' ('global' means every rank, 'local' means only the ranks within the same subdomain in data)
        '''
        if sync_loss not in ['global', 'local']:
            raise ValueError('sync_loss must be either "global" or "local".')

        if zero_grad:
            model.zero_grad()

        with torch.set_grad_enabled(compute_grad):
            if has_model_handler:
                outputs = model(inputs, chunks_amount=data_chunks_amount)
            else:
                outputs = model(inputs)

            if outputs_only:
                return [output for output in outputs]

        losses = [0] * data_chunks_amount if has_model_handler else []
        loss = torch.tensor(0.0, device=inputs.device)

        if has_model_handler and model.model_handler.is_last_stage():
            for i, out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = torch.tensor((sum(losses) / len(losses)).item(), device=model.tensor_device)
        elif not has_model_handler:
            loss = criterion(outputs, targets.to(outputs.device))

        # Distributed processing (only if model_handler is present)
        if has_model_handler:
            if sync_loss == 'global':
                if model.model_handler.is_last_stage():
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(mode='global'))
                    loss = loss / model.model_handler.tot_replicas
                last_ranks = model.model_handler.get_stage_ranks(stage_name='last', mode='global')
                loss_broadcast = dist.broadcast(loss.detach(), src=last_ranks[0], group=model.model_handler.global_model_group, async_op=True)
            else:
                if model.model_handler.is_last_stage():
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(mode='local'))
                    loss = loss / model.num_replicas_per_subdomain
                last_stage_ranks = model.model_handler.get_stage_ranks(stage_name='last', mode='local')
                if len(last_stage_ranks) > 1:
                    raise ValueError('Tensor sharding not implemented yet.')
                loss_broadcast = dist.broadcast(loss.detach(), src=last_stage_ranks[0], group=model.model_handler.get_sd_group(), async_op=True)
        elif not has_model_handler and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size() # gradient averaging taken care of by DDP, assuming model is wrapped by DDP

        # Compute gradients
        if compute_grad and torch.is_grad_enabled():
            if has_model_handler:
                model.backward(losses)
            else:
                loss.backward()
            
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip) 

        if has_model_handler:
            loss_broadcast.wait()

        if return_output:
            return loss.item(), [output for output in outputs] if has_model_handler else outputs

        return loss.item()

    return closure2


def decide_tensor_device(ws, backend, gpu_id):
    loc_rank = os.environ['LOCAL_RANK']
    if torch.cuda.is_available():
        if backend == 'gloo':
            if torch.cuda.device_count() < ws:
                return f'cuda:{gpu_id}'
            else:
                # Local rank
                return f'cuda:{loc_rank}'
        else:
            if gpu_id is None:
                gpu_id = loc_rank
            return f'cuda:{gpu_id}'
    else:
        return 'cpu'

def list_flattener(l):
    '''
    Flattens a list of lists of lists of ... to a single list.
    '''
    while any(isinstance(i, list) for i in l):
        l = [item for sublist in l for item in sublist]
    return l

def get_starting_info(rank, base_file_name, epoch_file_name, num_epochs):
    starting_epoch = 0
    starting_num_iters = 0
    starting_network = ""
    epoch_results = []
    iter_results = []

    # Check if the model has already been trained
    max_epoch_already_trained = -1
    saved_networks = os.listdir('../saved_networks')
    for saved_network in saved_networks:
        if base_file_name in saved_network:
            saved_network_epoch = int(
                saved_network.split('_epoch_')[1].split('.pth')[0])

            if saved_network_epoch > max_epoch_already_trained:
                max_epoch_already_trained = saved_network_epoch
                starting_network = saved_network

    # Check that the corresponding csv files exist
    if max_epoch_already_trained > -1:
        if os.path.exists(epoch_file_name):
            starting_epoch = max_epoch_already_trained + 1
            if starting_epoch > num_epochs+1:
                if rank == 0:
                    print("Model already fully trained. Exiting...")
                exit(0)
            # Load epoch results
            df = pd.read_csv(epoch_file_name)
            epoch_results = df.to_dict('records')
            # Load iteration results
            df = pd.read_csv(epoch_file_name.replace('.csv', '_iter.csv'))
            iter_results = df.to_dict('records')
            # Get the number of iterations
            starting_num_iters = iter_results[-1]['iteration']
            # Print details
            if rank == 0 and max_epoch_already_trained > -1:
                print(
                    f"Model with parameters {base_file_name} already trained for {max_epoch_already_trained} epochs")
        else:
            starting_network = ""

    return starting_epoch, starting_num_iters, epoch_results, iter_results, starting_network

