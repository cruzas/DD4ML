import math
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor


def _get_world_size_and_scale(config):
    """Helper to get world size and delta scaling factor."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    delta_scale = 1.0 / world_size if config.norm_type != math.inf else 1.0
    return world_size, delta_scale


def get_state_dict(model):
    """
    Returns the state dictionary of the model, handling both single and distributed models.
    If the model is wrapped in a DataParallel or DistributedDataParallel, it extracts the state_dict
    from the underlying module.
    """
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def get_apts_hparams(config):
    """
    Returns default hyperparameters for the Additively Preconditioned Trust-Region Strategy (APTS).
    """
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
    """
    Returns default hyperparameters for the L-S-SR1 Trust-Region optimizer.
    (Also when used as the global optimizer for APTS.)
    """
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
    """
    Returns default hyperparameters for the L-S-SR1 Trust-Region optimizer,
    when used as the local optimizer for APTS.
    """
    world_size, delta_scale = _get_world_size_and_scale(config)
    return {
        "delta": config.delta * delta_scale,
        "min_delta": config.min_delta * delta_scale,
        "max_delta": config.max_delta * delta_scale,  # Lower maximum for local updates
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
    """
    Returns default hyperparameters for the Trust-Region optimizer.
    (Also when used as the global optimizer for APTS.)
    """
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
    """
    Returns default hyperparameters for the Trust-Region optimizer,
    when used as the local optimizer for APTS.
    """
    world_size, delta_scale = _get_world_size_and_scale(config)
    return {
        "delta": config.delta * delta_scale,
        "max_delta": config.max_delta * delta_scale,  # Lower maximum for local updates
        "min_delta": config.min_delta * delta_scale,
        "nu": 0.45,
        "inc_factor": 1.5,  # Reduced increase factor
        "dec_factor": 0.6,
        "nu_dec": 0.3,
        "nu_inc": 0.7,
        "mem_length": 5,
        "norm_type": config.norm_type,
        "second_order": config.loc_second_order,
        "dogleg": config.loc_dogleg,
        "tol": config.tol,
    }


def get_loc_tradam_hparams(config):
    """
    Returns default hyperparameters for the TRAdam optimizer,
    when used as the local optimizer for APTS.
    """
    return {
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "norm_type": config.norm_type,
    }


def get_asntr_hparams(config):
    """
    Return default hyperparameters for the Adaptive Sub-sample Non-monotone Trust-Region (ASNTR) optimizer.
    """
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
    If grad_norm <= tol
        - returns zeros
    Else:
        - Closed-form first-order TR: step = -gradient * (trust_radius / grad_norm)
        - Predicted reduction = trust_radius * grad_norm
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
    obs_solver,  # exposes .solve_tr_subproblem(g, delta, gamma, Psi, Minv)
    tol: float,
    dogleg: bool = False,  # if True, use dogleg between Cauchy and OBS
) -> Tuple[torch.Tensor, float]:
    """
    TR via LSR1+OBS, with optional dogleg:
      1. If ||g|| <= tol, return zero
      2. Precompute LSR1 factors
      3. Build Cauchy point p_c on the model m(p) = g^T*p + 0.5 * p^T * B * p:
         alpha_c = (g^T * g)/(g^T * B * g),  p_c = -alpha_c g  (clipped to trust-region radius).
      4. Let p_b = OBS-solver's unconstrained minimizer.
      5. If dogleg:
           - If ||p_b|| <= delta, p = p_b
           - Elif ||p_c|| >= delta, p = p_c
           - Else find tau in [0,1] s.t. ||p_c + tau*(p_b-p_c)|| = delta and set
             p = p_c + tau*(p_b-p_c)
         Else:
           p = p_b
      6. pred ≔ -(g^T*p + 0.5*p^T*B*p).
    """
    if grad_norm <= tol:
        return torch.zeros_like(gradient), 0.0

    # Precompute and helpers
    lsr1_hessian.precompute()
    delta = trust_radius

    # OBS (full step)
    delta_tensor = torch.scalar_tensor(delta, device=gradient.device, dtype=gradient.dtype)
    p_b = obs_solver.solve_tr_subproblem(
        gradient,
        delta_tensor,
        lsr1_hessian.gamma,
        lsr1_hessian.Psi,
        lsr1_hessian.Minv,
    )

    # Cauchy point along -g
    Bg = lsr1_hessian.B(gradient)
    gBg = gradient.dot(Bg)
    # Safe guard it if g^T*B*g <= 0
    alpha = grad_norm**2 / gBg if gBg > 0 else 0.0
    p_cand = -gradient * alpha
    # Clip to trust-region radius
    if torch.norm(p_cand) >= delta:
        p_c = -gradient * (delta / grad_norm)
    else:
        p_c = p_cand

    # Dogleg combination
    if dogleg and gBg > 0:
        p_b_norm = torch.norm(p_b)
        p_c_norm = torch.norm(p_c)
        if p_b_norm <= delta:
            p = p_b
        elif p_c_norm >= delta:
            p = p_c
        else:
            # solve ||p_c + tau*(p_b-p_c)‖ = delta
            d = p_b - p_c
            a = d.dot(d)
            b = 2 * p_c.dot(d)
            c = p_c.dot(p_c) - delta**2
            disc = b**2 - 4 * a * c  # discriminant

            if disc < 0:
                p = p_c  # fallback to Cauchy point if no valid tau
            else:
                tau = (-b + torch.sqrt(torch.clamp(disc, min=0.0))) / (2 * a)
                p = p_c + tau * d
    else:
        p = p_b

    # Compute the predicted reduction
    g_dot_p = gradient.dot(p)
    p_B_p = p.dot(lsr1_hessian.B(p))
    predicted = -(g_dot_p + 0.5 * p_B_p)

    return p, predicted


def ensure_tensor(d, device):
    """
    Ensures that the input "d" is a tensor on the specified device.
    If "d" is an int or float, it converts it to a tensor.
    """
    if torch.is_tensor(d):
        return d.to(device)
    elif isinstance(d, (int, float)):
        return torch.scalar_tensor(d, device=device)
    else:
        raise TypeError(
            f"Unexpected type for d: {type(d)}. Expected int, float, or torch.Tensor."
        )
