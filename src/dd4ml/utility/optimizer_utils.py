import math
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor


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


def get_apts_hparams(config):
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
        "dogleg": config.glob_dogleg,
        "mem_length": config.mem_length,
        "max_wolfe_iters": config.max_wolfe_iters,
        "max_zoom_iters": config.max_zoom_iters,
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
        "c_1": 1e-4,
        "c_2": 0.9,
        "alpha_max": 2.0,
        "sync": True,
        "paper_tr_update": config.paper_tr_update,
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
        "dogleg": config.loc_dogleg,
        "mem_length": config.mem_length,
        "max_wolfe_iters": config.max_wolfe_iters,
        "max_zoom_iters": config.max_zoom_iters,
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
        "c_1": 1e-4,
        "c_2": 0.9,
        "alpha_max": 2.0,
        "sync": False,
        "paper_tr_update": config.paper_tr_update,
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
        "dogleg": config.glob_dogleg,
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
        "dogleg": config.loc_dogleg,
        "tol": config.tol,
    }


def get_asntr_hparams(config):
    """Return default hyperparameters for the ASNTR optimizer."""
    return {
        "device": config.device if hasattr(config, "device") else "cpu",
        "learning_rate": (
            config.learning_rate if hasattr(config, "learning_rate") else 0.01
        ),
        "delta": config.delta,
        "max_delta": config.max_delta,
        "gamma": 1e-3,
        "second_order": config.glob_second_order,
        "dogleg": config.glob_dogleg,
        "mem_length": config.mem_length,
        "eta": 1e-4,
        "nu": 1e-4,
        "eta_1": 0.1,
        "eta_2": 0.75,
        "tau_1": 0.5,
        "tau_2": 0.8,
        "tau_3": 2.0,
        "norm_type": config.norm_type,
        "c_1": 1.0,
        "c_2": 100,
        "alpha": 1.1,
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
    """
    Closed-form first-order TR (steepest-descent on ball).
    If ‖g‖ <= tol, returns zero; else p = -(delta/||g||)*g.
    Predicted reduction = delta * ||g||.
    """

    if grad_norm <= tol:
        return torch.zeros_like(gradient), 0.0

    step = -gradient * (trust_radius / grad_norm)
    predicted = trust_radius * grad_norm
    return step, predicted


def solve_tr_second_order(
    gradient: torch.Tensor,
    grad_norm: float,
    trust_radius: float,
    lsr1_hessian,  # exposes .precompute(), .B(v), .gamma, .Psi, .Minv
    obs_solver,  # exposes .solve_tr_subproblem(g, δ, γ, Ψ, Minv)
    tol: float,
    dogleg: bool = False,  # if True, use dogleg between Cauchy and OBS
) -> Tuple[torch.Tensor, float]:
    """
    TR via LSR1+OBS, with optional dogleg:
      1. If ||g|| <= tol, return zero.
      2. Precompute LSR1 factors.
      3. Build Cauchy point p_c on the model m(p) = gᵀp + ½ pᵀ B p:
         alpha_c = (gᵀg)/(gᵀBg),  p_c = -alpha_c g  (clipped to ball).
      4. Let p_b = OBS-solver's unconstrained minimiser.
      5. If dogleg:
           • if ‖p_b‖ ≤ δ, p = p_b
           • elif ‖p_c‖ ≥ δ, p = p_c
           • else find τ∈[0,1] s.t. ‖p_c + τ(p_b-p_c)‖ = δ and set
             p = p_c + τ (p_b-p_c)
         else:
           p = p_b
      6. pred ≔ -(gᵀp + ½ pᵀ B p).
    """
    if grad_norm <= tol:
        return torch.zeros_like(gradient), 0.0

    # 1. Precompute and helpers
    lsr1_hessian.precompute()
    B = lambda v: lsr1_hessian.B(v)
    delta = trust_radius

    # 2. OBS (full step)
    p_b = obs_solver.solve_tr_subproblem(
        gradient,
        torch.tensor(delta, device=gradient.device, dtype=gradient.dtype),
        lsr1_hessian.gamma,
        lsr1_hessian.Psi,
        lsr1_hessian.Minv,
    )

    # 3. Cauchy point along -g
    Bg = B(gradient)
    gBg = gradient.dot(Bg)
    # safe guard if gᵀBg ≤ 0
    alpha = grad_norm**2 / gBg if gBg > 0 else 0.0
    p_cand = -gradient * alpha
    # clip to ball
    if torch.norm(p_cand) >= delta:
        p_c = -gradient * (delta / grad_norm)
    else:
        p_c = p_cand

    # 4. Dogleg combination
    if dogleg:
        if torch.norm(p_b) <= delta:
            p = p_b
        elif torch.norm(p_c) >= delta:
            p = p_c
        else:
            # solve ‖p_c + τ(p_b-p_c)‖ = δ
            d = p_b - p_c
            a = d.dot(d)
            b = 2 * p_c.dot(d)
            c = p_c.dot(p_c) - delta**2
            tau = (-b + torch.sqrt(b * b - 4 * a * c)) / (2 * a)
            p = p_c + tau * d
    else:
        p = p_b

    # 5. Predicted reduction
    g_dot_p = gradient.dot(p)
    p_B_p = p.dot(B(p))
    predicted = -(g_dot_p + 0.5 * p_B_p)

    return p, predicted


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
