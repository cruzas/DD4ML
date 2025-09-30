import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .optimizer_utils import get_state_dict


def _get_device():
    """Cache device to avoid repeated CUDA checks."""
    from .utils import get_default_device
    return get_default_device()


def flatten_params(model, out=None):
    # Infer param dtype/device from the first parameter
    first = next(model.parameters(), None)
    if first is None:
        # Empty model: honour out if provided, else default dtype/device
        if out is None:
            return torch.empty(0, dtype=torch.get_default_dtype())
        return out.narrow(0, 0, 0).clone()

    param_dtype = first.dtype
    param_device = first.device

    # Decide target dtype/device dynamically
    # Promote to float64 iff either params or out are float64
    if out is not None and out.dtype == torch.float64 or param_dtype == torch.float64:
        target_dtype = torch.float64
    else:
        target_dtype = param_dtype
    target_device = param_device

    if out is None:
        # Create flattened vector and upcast only if necessary.
        return parameters_to_vector(model.parameters()).to(
            dtype=target_dtype, device=target_device
        )

    # Conform out to target dtype/device (no pre-scan).
    if out.dtype != target_dtype or out.device != target_device:
        out = out.to(device=target_device, dtype=target_dtype)

    # Write flattened parameters into the preallocated tensor.
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        # Convert on the fly only if needed (avoids an upfront pass).
        src = p.data.view(-1)
        if src.dtype != target_dtype:
            src = src.to(target_dtype)
        out[offset : offset + numel].copy_(src)
        offset += numel

    return out.clone()


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
    # Preserve both device and dtype from original model
    original_param = next(model.parameters())
    new_model = new_model.to(device=original_param.device, dtype=original_param.dtype)
    new_model.load_state_dict(get_state_dict(base_model))
    return new_model


def broadcast_shuffle(num_layers: int, rank: int, world_size: int) -> torch.Tensor:
    device = _get_device()
    if rank == 0:
        perm = torch.randperm(num_layers, dtype=torch.long, device=device)
    else:
        perm = torch.empty(num_layers, dtype=torch.long, device=device)
    dist.broadcast(perm, src=0)
    return perm


def split_indices(perm: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    n = len(perm)
    q, r = divmod(n, world_size)
    start = rank * q + min(rank, r)
    end = start + q + (1 if rank < r else 0)
    return perm[start:end]


def mark_trainable(model: nn.Module):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    params = list(model.parameters())
    num_layers = len(params)

    if num_layers == 0:
        return model

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
    Returns a flat vector containing all trainable parameters of the model.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return torch.tensor([], dtype=torch.float32)

    return torch.cat([p.detach().view(-1) for p in trainable_params]).clone()


def trainable_grads_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Returns a flat vector containing the gradients of all trainable parameters of the model.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return torch.tensor([], dtype=torch.float32)

    grads = [p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
             for p in trainable_params]
    return torch.cat(grads).detach()


