import math
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor


class Timer:
    def __init__(self, timings, key):
        self.timings = timings
        self.key = key

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.timings[self.key] += time.time() - self.start


def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def get_apts_params(config):
    return {
        "delta": config.delta,
        "min_delta": config.min_delta,
        "nu_dec": 0.25,
        "nu_inc": 0.75,
        "inc_factor": 1.2,
        "dec_factor": 0.9,
        "max_delta": config.max_delta,
    }


def get_lssr1_tr_hparams(config):
    return {
        "delta": config.delta,
        "min_delta": config.min_delta,
        "max_delta": config.max_delta,
        "gamma": 1e-3,
        "second_order": config.glob_second_order,
        "mem_length": config.mem_length,
        "max_wolfe_iters": config.max_wolfe_iters,
        "mu": 0.9,
        "tau_1": 0.1,
        "tau_2": 0.25,
        "tau_3": 0.75,
        "nu_1": 0.25,
        "nu_2": 0.5,
        "nu_3": 0.9,
        "nu_4": 1.2,
        "tol": config.tol,
        "norm_type": config.norm_type,
        "sync": True,
    }


def get_lssr1_loc_tr_hparams(config):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    norm_type = config.norm_type
    delta_scale = 1.0 / world_size if config.norm_type != math.inf else 1.0
    delta = config.delta * delta_scale
    return {
        "delta": delta,
        "min_delta": config.min_delta,
        "max_delta": config.max_delta,  # Lower maximum for local updates
        "gamma": 1e-3,
        "second_order": config.loc_second_order,
        "mem_length": config.mem_length,
        "max_wolfe_iters": config.max_wolfe_iters,
        "mu": 0.9,
        "tau_1": 0.1,
        "tau_2": 0.25,
        "tau_3": 0.75,
        "nu_1": 0.25,
        "nu_2": 0.5,
        "nu_3": 0.9,
        "nu_4": 1.2,
        "tol": config.tol,
        "norm_type": config.norm_type,
        "sync": False,
    }


def get_tr_hparams(config):
    return {
        "delta": config.delta,
        "max_delta": config.max_delta,
        "min_delta": config.min_delta,
        "nu": 0.5,
        "inc_factor": 1.2,
        "dec_factor": 0.9,
        "nu_dec": 0.25,
        "nu_inc": 0.75,
        "mem_length": 5,
        "norm_type": config.norm_type,
        "second_order": config.glob_second_order,
        "tol": config.tol,
    }


def get_loc_tr_hparams(config):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    norm_type = config.norm_type
    delta_scale = 1.0 / world_size if config.norm_type != math.inf else 1.0
    delta = config.delta * delta_scale
    return {
        "delta": delta,
        "max_delta": config.max_delta,  # Lower maximum for local updates
        "min_delta": 1e-6,
        "nu": (
            0.45 if norm_type != math.inf else 0.5
        ),  # Adjusted to be more conservative locally
        "inc_factor": 1.5 if norm_type != math.inf else 1.2,  # Reduced increase factor
        "dec_factor": (
            0.6 if norm_type != math.inf else 0.9
        ),  # Slightly more aggressive reduction
        "nu_dec": 0.3 if norm_type != math.inf else 0.25,
        "nu_inc": 0.7 if norm_type != math.inf else 0.75,
        "mem_length": 5,
        "norm_type": config.norm_type,
        "second_order": config.loc_second_order,
        "tol": config.tol,
    }


def solve_tr_first_order(
    gradient: Tensor,
    grad_norm: float,
    trust_radius: float,
    tol: float,
) -> Tuple[Tensor, float]:
    """
    Closed-form first-order TR: step = -gradient * (trust_radius / grad_norm).
    Predicted reduction = trust_radius * grad_norm. If grad_norm <= tol, returns zeros.
    """
    if grad_norm <= tol:
        return torch.zeros_like(gradient), 0.0
    step = -gradient * (trust_radius / grad_norm)
    predicted = trust_radius * grad_norm
    return step, predicted


def solve_tr_second_order(
    gradient: Tensor,
    grad_norm: float,
    trust_radius: float,
    lsr1_hessian,  # an object exposing .precompute(), .gamma, .Psi, .Minv, .B(v)
    obs_solver,  # an object exposing .solve_tr_subproblem(g, delta, γ, Ψ, Minv)
    tol: float,
) -> Tuple[Tensor, float]:
    """
    TR via LSR1+OBS:
    - If grad_norm <= tol, returns zeros.
    - Otherwise calls lsr1_hessian.precompute(), then obs_solver.solve_tr_subproblem(...)
    - Computes predicted reduction = -(gᵀp + 0.5 pᵀ B p).
    """
    if grad_norm <= tol:
        return torch.zeros_like(gradient), 0.0

    # (Re)compute any LSR1 factors
    lsr1_hessian.precompute()
    delta = torch.tensor(trust_radius, device=gradient.device, dtype=gradient.dtype)
    p = -obs_solver.solve_tr_subproblem(
        gradient, delta, lsr1_hessian.gamma, lsr1_hessian.Psi, lsr1_hessian.Minv
    )
    # predicted reduction = -(gᵀp + 0.5 pᵀ B p)
    g_dot_p = torch.dot(gradient, p)
    p_B_p = torch.dot(p, lsr1_hessian.B(p))
    predicted = -(g_dot_p + 0.5 * p_B_p)
    return p, predicted.item()


def ensure_tensor(d, device):
    if isinstance(d, (int, float)):
        d = torch.tensor(d, device=device)
        return d
    elif torch.is_tensor(d):
        d = d.to(device)
        return d
    else:
        raise TypeError(
            f"Unexpected type for d: {type(d)}. Expected int, float, or torch.Tensor."
        )
