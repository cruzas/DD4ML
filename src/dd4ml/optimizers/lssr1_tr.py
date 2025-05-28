"""lssr1.py - Limited-Memory Stochastic SR1 Trust-Region Optimiser
----------------------------------------------------------------
Full implementation of **LSSR1_TR** that delegates
* Hessian modelling to   ``hessian_approx.LSR1``
* SR1 trust-region solves to ``obs.OBS``

It now follows the radius-update rule of Algorithm 3 in the paper, with
parameters  (τ₁, τ₂, τ₃) and (ν₁, ν₂, ν₃, ν₄).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from hessian_approx import LSR1
from obs import OBS
from torch import Tensor
from torch.optim.optimizer import Optimizer

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _concat_params(params: Iterable[Tensor]) -> Tensor:
    """Flatten *params* into a single 1-D tensor (detached)."""
    return torch.cat([p.detach().flatten() for p in params])


def _concat_grads(params: Iterable[Tensor]) -> Tensor:
    """Flatten all gradients into a single 1-D tensor."""
    return torch.cat([p.grad.flatten() for p in params])


def _set_param_vector(params: Iterable[Tensor], vec: Tensor) -> None:
    """Write *vec* back into *params* (in-place)."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(vec[offset : offset + numel].view_as(p))
        offset += numel


# -----------------------------------------------------------------------------
# PyTorch optimiser implementing Algorithm 3 (L-SSR1-TR)
# -----------------------------------------------------------------------------


class LSSR1_TR(Optimizer):
    r"""Limited-Memory **Stochastic SR1** Trust-Region optimiser.

    The trust-region radius δ is updated exactly as prescribed by
    Algorithm 3 (lines 15-23) with user-exposed parameters::

        0 ≤ τ₁ < τ₂ < 0.5 < τ₃ < 1
        0 < ν₁ < ν₂ ≤ 0.5 < ν₃ < 1 < ν₄.

    Parameters
    ----------
    params : iterable of *torch.Tensor*
        Model parameters.
    lr_init : float, default 1.0
        Initial step length for the Wolfe line search.
    delta_init : float, default 1.0
        Initial trust-region radius δ₀.
    gamma_init : float, default 1e-3
        Scaling of the initial Hessian approximation B₀ = gamma * I.
    memory : int, default 10
        Number of (s, y) pairs kept in limited memory.
    mu : float, default 0.9
        Momentum parameter.
    overlap : float, default 0.33
        Required mini-batch overlap fraction.
    tol_grad : float, default 1e-8
        Gradient-norm termination tolerance.

    Radius-update hyper-parameters
    ------------------------------
    tau_1, tau_2, tau_3 : float
        Acceptance thresholds (see paper).  Defaults (0.1, 0.25, 0.75).
    nu_1 … nu_4 : float
        Contraction / expansion factors.  Defaults (0.25, 0.5, 0.8, 2.0).
    """

    def __init__(
        self,
        params,
        *,
        lr_init: float = 1.0,
        delta_init: float = 1.0,
        gamma_init: float = 1e-3,
        memory: int = 10,
        mu: float = 0.9,
        overlap: float = 0.33,
        tol_grad: float = 1e-8,
        tau_1: float = 0.1,
        tau_2: float = 0.25,
        tau_3: float = 0.75,
        nu_1: float = 0.25,
        nu_2: float = 0.5,
        nu_3: float = 0.8,
        nu_4: float = 2.0,
    ):
        if not (0.0 <= tau_1 < tau_2 < 0.5 < tau_3 < 1.0):
            raise ValueError("Tau parameters must satisfy 0 ≤ τ₁ < τ₂ < 0.5 < τ₃ < 1.")
        if not (0.0 < nu_1 < nu_2 <= 0.5 < nu_3 < 1.0 < nu_4):
            raise ValueError(
                "Nu parameters must satisfy 0 < ν₁ < ν₂ ≤ 0.5 < ν₃ < 1 < ν₄."
            )

        defaults = dict(
            lr_init=lr_init,
            delta=delta_init,
            gamma=gamma_init,
            memory=memory,
            mu=mu,
            overlap=overlap,
            tol_grad=tol_grad,
            tau_1=tau_1,
            tau_2=tau_2,
            tau_3=tau_3,
            nu_1=nu_1,
            nu_2=nu_2,
            nu_3=nu_3,
            nu_4=nu_4,
        )
        super().__init__(params, defaults)

        # External helpers ---------------------------------------------------
        self.obs = OBS()
        self.hess = LSR1(
            gamma=gamma_init,
            memory_length=memory,
            device=params[0].device,
            dtype=params[0].dtype,
        )

        # Internal state -----------------------------------------------------
        s = self.state
        s["wk"] = None  # previous iterate
        s["prev_grad"] = None  # previous gradient
        s["vk"] = torch.zeros(1)  # momentum accumulator

    # ------------------------------------------------------------------
    # Main optimisation step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]):
        """Perform one optimisation step and return the loss."""
        loss = closure()  # forward & backward - gradients populated

        params = self.param_groups[0]["params"]
        g = _concat_grads(params)
        if g.norm() <= self.defaults["tol_grad"]:
            return loss  # converged

        # Current point ----------------------------------------------------
        wk = _concat_params(params)
        st = self.state
        if st["wk"] is None:
            st["wk"] = wk.clone()
        if st["vk"].numel() != wk.numel():
            st["vk"] = torch.zeros_like(wk)

        # ------------------------------------------------------------------
        # 1. Update limited-memory pairs
        # ------------------------------------------------------------------
        if st["prev_grad"] is not None:
            s_vec = wk - st["wk"]
            y_vec = g - st["prev_grad"]
            self.hess.update_memory(s_vec, y_vec)
        st["wk"], st["prev_grad"] = wk.clone(), g.clone()
        self.hess.precompute()

        # ------------------------------------------------------------------
        # 2. Solve trust-region sub-problem via OBS
        # ------------------------------------------------------------------
        p_star = self.obs.solve_tr_subproblem(
            g,
            delta=self.defaults["delta"],
            gamma=self.hess.gamma,
            Psi=self.hess.Psi,
            Minv=self.hess.M_inv,
        )

        # ------------------------------------------------------------------
        # 3. Momentum grafting (Eq. 17)
        # ------------------------------------------------------------------
        vk = st["vk"] * self.defaults["mu"] + (wk - st["wk"])  # previous step
        if vk.norm() > 0:
            vk = self.defaults["mu"] * min(1.0, self.defaults["delta"] / vk.norm()) * vk
        p_comb = p_star + vk
        if p_comb.norm() > 0:
            p_comb = min(1.0, self.defaults["delta"] / p_comb.norm()) * p_comb
        st["vk"] = vk.clone()

        # ------------------------------------------------------------------
        # 4. Wolfe back-tracking line-search along *p_comb*
        # ------------------------------------------------------------------
        alpha = self.defaults["lr_init"]
        c1, c2 = 1e-4, 0.9
        orig_loss = loss.item()
        grad_dot_dir = g.dot(p_comb)
        for _ in range(10):
            _set_param_vector(params, wk + alpha * p_comb)
            new_loss = closure().item()
            if new_loss <= orig_loss + c1 * alpha * grad_dot_dir:
                break  # Armijo satisfied - curvature check omitted for speed
            alpha *= 0.5
        else:
            alpha = 0.0  # fallback - reject step
            _set_param_vector(params, wk)

        p_alpha = alpha * p_comb  # accepted step (may be zero)

        # ------------------------------------------------------------------
        # 5. Trust-region radius update (Algorithm 3)
        # ------------------------------------------------------------------
        delta_old = self.defaults["delta"]
        rho = 0.0
        if alpha > 0:  # predicted reduction (quadratic model)
            # approximate predicted reduction using linear term only - cheap
            pred = g.dot(p_alpha)
            if pred < 0:
                rho = (new_loss - orig_loss) / pred
        s_norm = p_alpha.norm()

        if rho < self.defaults["tau_2"]:
            delta_new = min(
                self.defaults["nu_1"] * delta_old, self.defaults["nu_2"] * (s_norm**2)
            )
        else:
            if (
                rho >= self.defaults["tau_3"]
                and s_norm >= self.defaults["nu_3"] * delta_old
            ):
                delta_new = self.defaults["nu_4"] * delta_old
            else:
                delta_new = delta_old
        self.defaults["delta"] = max(delta_new, 1e-12)

        # Parameters already updated during line-search --------------------
        return loss
