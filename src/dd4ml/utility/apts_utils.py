import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .optimizer_utils import get_state_dict


def fix_aggregated_local_steps_pnorm(
    aggregated_step, global_grad, p=2.0, tol=1e-6, max_iter=50
):
    """
    Projects aggregated_step onto the direction of global_grad under the p-norm.

    Cases:
      - p = 2: Uses closed-form projection.
      - p = 1: Uses weighted median of ratios a_i/g_i.
      - p = float('inf'): Minimizes f(alpha)=max_i|a_i - alpha*g_i| via ternary search.
      - Otherwise (p > 1, p ≠ 2): Finds optimal alpha via derivative-based bisection.
    """
    # Return zero if global_grad is zero.
    if torch.norm(global_grad) == 0:
        return torch.zeros_like(global_grad)

    # Special case: p = 2
    if abs(p - 2.0) < tol:
        norm_global_sq = torch.dot(global_grad, global_grad)
        alpha = (
            torch.dot(aggregated_step, global_grad) / norm_global_sq
            if norm_global_sq > 0
            else 0.0
        )
        return alpha * global_grad

    # Special case: p = 1 (minimize L1 norm)
    elif abs(p - 1.0) < tol:
        valid = global_grad != 0
        if valid.sum() == 0:
            return torch.zeros_like(global_grad)
        # Compute ratios and weights for indices with nonzero global_grad.
        ratios = (aggregated_step[valid] / global_grad[valid]).detach().cpu().numpy()
        weights = torch.abs(global_grad[valid]).detach().cpu().numpy()
        sorted_indices = ratios.argsort()
        ratios_sorted = ratios[sorted_indices]
        weights_sorted = weights[sorted_indices]
        total_weight = weights_sorted.sum()
        cumulative = 0.0
        for r, w in zip(ratios_sorted, weights_sorted):
            cumulative += w
            if cumulative >= total_weight / 2:
                alpha = r
                break
        return alpha * global_grad

    # Special case: p = infinity (minimize L-inf norm)
    elif p == float("inf"):
        valid = global_grad != 0
        if valid.sum() == 0:
            return torch.zeros_like(global_grad)
        # Use ratios from valid indices to bracket the optimum.
        ratios = (aggregated_step[valid] / global_grad[valid]).detach().cpu().numpy()
        L, R = ratios.min(), ratios.max()

        def f(alpha):
            diff = aggregated_step - alpha * global_grad
            return torch.max(torch.abs(diff)).item()

        for _ in range(max_iter):
            m1 = L + (R - L) / 3
            m2 = R - (R - L) / 3
            if f(m1) > f(m2):
                L = m1
            else:
                R = m2
        alpha = (L + R) / 2
        return alpha * global_grad

    # General case: p > 1 (and not 1,2,inf)
    else:

        def derivative(alpha):
            diff = aggregated_step - alpha * global_grad
            # Derivative of f(alpha)=||aggregated_step-alpha*global_grad||_p^p (up to constant factor)
            return torch.sum(
                global_grad * torch.sign(diff) * torch.abs(diff) ** (p - 1)
            )

        d0 = derivative(0.0)
        if abs(d0) < tol:
            return torch.zeros_like(global_grad)

        if d0 > 0:
            alpha_high = 0.0
            alpha_low = -1.0
            while derivative(alpha_low) > 0 and abs(alpha_low) < 1e6:
                alpha_low *= 2.0
        else:
            alpha_low = 0.0
            alpha_high = 1.0
            while derivative(alpha_high) < 0 and abs(alpha_high) < 1e6:
                alpha_high *= 2.0

        for _ in range(max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2.0
            d_mid = derivative(alpha_mid)
            if abs(d_mid) < tol:
                return alpha_mid * global_grad
            if d_mid > 0:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
        return ((alpha_low + alpha_high) / 2.0) * global_grad


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

    local_indices = split_indices(perm=perm, rank=rank, world_size=world_size)

    # Store and apply
    model._trainable_indices = local_indices
    for i, p in enumerate(params):
        p.requires_grad = i in local_indices
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
