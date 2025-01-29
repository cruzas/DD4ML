import os

import pandas as pd
import torch
import torch.distributed as dist


def cross_entropy_transformers(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False, data_chunks_amount=1, grad_norm_clip=None, outputs_only=False):
    """
    NOTE: Losses from different chunks are averaged.
    """
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.model_handler.is_last_stage() and targets is not None and not outputs_only:
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
            outputs = model(inputs, chunks_amount=data_chunks_amount)
            if outputs_only:
                return [output for output in outputs]
        losses = [0] * data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.model_handler.is_last_stage():
            for i, out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = torch.tensor(
                (sum(losses)/len(losses)).item()).to(model.tensor_device)
        # Average losses across replicas
        if sync_loss == 'global':
            if model.model_handler.is_last_stage():
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(
                    mode='global'))  # Summing the losses across final layers of each replicas
                loss = loss/model.model_handler.tot_replicas
            last_ranks = model.model_handler.get_stage_ranks(
                stage_name='last', mode='global')
            # each replica gets the average loss across all replicas (since we are averaging the losses first)
            loss_broadcast = dist.broadcast(tensor=loss.detach(
            ), src=last_ranks[0], group=model.model_handler.global_model_group, async_op=True)
            # -> all subdomains will have the same loss
        else:
            if model.model_handler.is_last_stage():
                # Summing the losses across shard 0 of final layers of each replicas within the same subdomain
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM,
                                group=model.model_handler.get_layers_copy_group(mode='local'))
                loss = loss/model.num_replicas_per_subdomain
            last_stage_ranks = model.model_handler.get_stage_ranks(
                stage_name='last', mode='local')
            if len(last_stage_ranks) > 1:
                raise ValueError('Tensor sharding not implemented yet.')
            # shard 0 of last layer of first model replica broadcasts the loss to all other replicas within the same subdomain
            loss_broadcast = dist.broadcast(tensor=loss.detach(
            ), src=last_stage_ranks[0], group=model.model_handler.get_sd_group(), async_op=True)
            # -> each subdomains may have a different loss

        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip) 
        loss_broadcast.wait()
        if return_output:
            if model.model_handler.is_last_stage():
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output for output in outputs]
            else:
                return loss.item(), None
        return loss.item()
    return closure2

def decide_tensor_device(ws, backend, gpu_id):
    if torch.cuda.is_available():
        if backend == 'gloo':
            if torch.cuda.device_count() < ws:
                return f'cuda:{gpu_id}'
            else:
                return f'cuda:{dist.get_rank()}'
        else:
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

