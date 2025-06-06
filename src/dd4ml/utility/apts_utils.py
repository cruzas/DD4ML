import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .optimizer_utils import get_state_dict


def flatten_params(model, out=None):
    if out is None:
        return parameters_to_vector(model.parameters())
    # Write flattened parameters into a preallocated tensor.
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        out[offset : offset + numel].copy_(param.data.view(-1))
        offset += numel
    return out


def restore_params(model, flat_params):
    vector_to_parameters(flat_params, model.parameters())

@torch.no_grad()
def apts_ip_restore_params(model, flat_params):
    for i, p in enumerate(model.parameters()):
        p.copy_(flat_params.tensor[i])

def clone_model(model):
    base_model = model.module if hasattr(model, "module") else model
    config_copy = copy.deepcopy(base_model.config)
    config_copy.model_type = None
    new_model = type(base_model)(config_copy)
    new_model.load_state_dict(get_state_dict(base_model))
    return new_model.to(next(model.parameters()).device)


def broadcast_shuffle(num_layers: int, rank: int, world_size: int) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        perm = torch.randperm(num_layers, dtype=torch.long, device=device)
    else:
        perm = torch.empty(num_layers, dtype=torch.long, device=device)
    dist.broadcast(perm, src=0)
    return perm


def split_indices(perm: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    # q, r = divmod(len(perm), world_size)
    # start = rank * q + min(rank, r)
    # end = start + q + (1 if rank < r else 0)
    # return perm[start:end]
    n = len(perm)
    start, remaining = 0, n
    for i in range(world_size):
        size = remaining // (world_size - i) if i < world_size - 1 else remaining
        if i == rank:
            return perm[start : start + size]
        start += size
        remaining -= size


def mark_trainable(model: nn.Module):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    params = list(model.parameters())
    num_layers = len(params)

    if world_size > num_layers:
        raise ValueError(
            "World size cannot be greater than the number of layers in the model."
        )

    # All ranks must take part in the broadcast so still call it unconditionally
    perm = broadcast_shuffle(num_layers=num_layers, rank=rank, world_size=world_size)

    loc_indices = split_indices(perm=perm, rank=rank, world_size=world_size)

    # Store and apply
    model._trainable_indices = loc_indices
    for i, p in enumerate(params):
        p.requires_grad = i in loc_indices
    return model


def print_params_norm(model: nn.Module):
    with torch.no_grad():
        # Flatten the parameters and print the norm
        flat_params = flatten_params(model)
        norm = torch.norm(flat_params)
        print(norm.item())


def trainable_params_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Returns a vector containing only the trainable parameters of the model.
    """
    return torch.cat(
        [
            (p.grad.view(-1) if p.grad is not None else p.new_zeros(p.numel()))
            for p in model.parameters()
        ]
    ).detach()
