"""trust_region_second_order.py
--------------------------------
Second-order (OBS + limited-memory SR1) **or** first-order (gradient-only)
trust-region optimiser for PyTorch.  The behaviour is controlled by a single
boolean flag `second_order` passed at construction time.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch.optim import Optimizer

from dd4ml.optimizers.hessian_approx import LSR1
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_trust_region_params

__all__ = ["TR"]


def _flat_grad(model) -> torch.Tensor:
    """Return the concatenated gradients of all parameters."""
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads) if grads else torch.zeros(0, dtype=torch.float32)


def _flat_params(model) -> torch.Tensor:
    """Return parameters flattened into a single tensor (no gradients)."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])


class TR(Optimizer):
    """Limited-memory SR1 / OBS trust-region optimiser.

    Parameters
    ----------
    model          : nn.Module   - model whose parameters are optimised.
    second_order   : bool        - if *False* the optimiser falls back to a
                                   first-order Cauchy step inside the
                                   trust-region; if *True* it uses OBS with an
                                   SR1 Hessian approximation.
    Remaining arguments are identical to the previous version and documented in
    the docstring below.
    """

    # ---------------------------------------------------------------------
    # Static helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def setup_TR_args(config):
        params = get_trust_region_params(config)
        for k, v in params.items():
            setattr(config, k, v)
        return config

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        model,
        lr: float = 0.01,
        max_lr: float = 1.0,
        min_lr: float = 1e-4,
        nu: float = 0.5,
        inc_factor: float = 2.0,
        dec_factor: float = 0.5,
        nu_dec: float = 0.25,
        nu_inc: float = 0.75,
        max_iter: int = 10,
        norm_type: int = 2,
        *,
        second_order: bool = True,
    ) -> None:
        super().__init__(model.parameters(), {"lr": lr})
        self.model = model
        self.param_list = list(model.parameters())

        # trust‑region radii / factors
        self.lr = float(lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.inc_factor = float(inc_factor)
        self.dec_factor = float(dec_factor)
        self.nu_dec = float(nu_dec)
        self.nu_inc = float(nu_inc)
        self.nu = min(nu, nu_dec)
        self.norm_type = norm_type

        # memory size for (s,y) pairs
        self.max_iter = int(max_iter)

        # first‑ vs second‑order toggle
        self.second_order = bool(second_order)

        # ------------------------------------------------------------------
        # Second‑order helpers (allocated only if needed)
        # ------------------------------------------------------------------
        if self.second_order:
            device = next(model.parameters()).device
            self.hess = LSR1(
                gamma=1.0, memory_length=max_iter, device=device, tol=1e-10
            )
            self.obs = OBS()
        else:
            self.hess = None  # type: ignore
            self.obs = None  # type: ignore

        # Persistent state
        self.state["prev_loss"]: Optional[float] = None
        self.state["prev_grad"]: Optional[torch.Tensor] = None
        self.state["prev_params"]: Optional[torch.Tensor] = None
        self.state["initialized"] = False

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _apply_update(self, step: torch.Tensor) -> None:
        """Add the flattened *step* vector to the model parameters."""
        offset = 0
        for p in self.param_list:
            num = p.numel()
            p.data.add_(step[offset : offset + num].view_as(p.data))
            offset += num

    # ------------------------- sub‑problem solvers -----------------------
    def _solve_tr_first_order(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        """Cauchy step inside the trust region (gradient only)."""
        g_norm = torch.norm(g, p=self.norm_type)
        if g_norm == 0:
            return torch.zeros_like(g), 0.0
        step = -g * (delta / g_norm)
        predicted = delta * g_norm  # |g|*delta, since B ≈ I and step = −ĝ·δ
        return step, float(predicted)

    def _solve_tr_second_order(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        """OBS closed‑form solution using the current SR1 approximation."""
        # build Ψ, M⁻¹, γ
        self.hess.precompute()
        # OBS returns *minus* the step; we keep the sign convention consistent
        p_star = -self.obs.solve_tr_subproblem(
            g,
            torch.tensor(delta, device=g.device, dtype=g.dtype),
            self.hess.gamma,
            self.hess.Psi,
            self.hess.M_inv,
        )
        # model reduction  m(p) = gᵀp + 0.5 pᵀBp
        gTp = torch.dot(g, p_star)
        pTBp = torch.dot(p_star, self.hess.B(p_star))
        predicted = -(gTp + 0.5 * pTBp)
        return p_star, float(predicted)

    # ------------------------------------------------------------------
    # Main optimisation step
    # ------------------------------------------------------------------
    def step(self, closure, **_) -> float:
        """Perform one trust-region iteration.  *closure* must recompute the
        loss and gradients when called with *compute_grad=True*.
        """
        # 1. evaluate objective and gradient at x_k
        loss_tensor = closure(compute_grad=True)
        loss_val = (
            float(loss_tensor)
            if isinstance(loss_tensor, (float, int))
            else float(loss_tensor.item())
        )
        grad = _flat_grad(self.model)
        if grad.numel() == 0:
            return loss_val  # nothing to do

        # initialise persistent state the first time we are called
        if not self.state["initialized"]:
            self.state.update(
                {
                    "prev_loss": loss_val,
                    "prev_grad": grad.clone(),
                    "prev_params": _flat_params(self.model),
                    "initialized": True,
                }
            )

        # 2. solve TR sub‑problem
        if self.second_order and len(self.hess._S) > 0:
            step_vec, pred_red = self._solve_tr_second_order(grad, self.lr)
        elif self.second_order:  # no curvature pairs yet → fall back
            step_vec, pred_red = self._solve_tr_first_order(grad, self.lr)
        else:
            step_vec, pred_red = self._solve_tr_first_order(grad, self.lr)

        # keep copy for rollback
        params_before = _flat_params(self.model)

        # 3. apply candidate step
        self._apply_update(step_vec)

        # 4. evaluate at trial point x_k + s
        new_loss_tensor = closure(compute_grad=True)
        new_loss_val = (
            float(new_loss_tensor)
            if isinstance(new_loss_tensor, (float, int))
            else float(new_loss_tensor.item())
        )

        # 5. acceptance test
        act_red = self.state["prev_loss"] - new_loss_val
        rho = act_red / (pred_red + 1e-12)

        if act_red > 0 and rho >= self.nu_dec:  # accept
            accepted = True
            if rho >= self.nu_inc and torch.norm(step_vec) >= 0.9 * self.lr:
                self.lr = min(self.max_lr, self.inc_factor * self.lr)
            # SR1 memory update (only if second‑order)
            if self.second_order:
                y = _flat_grad(self.model) - grad
                self.hess.update_memory(step_vec.clone(), y.clone())
        else:  # reject ⇒ rollback
            accepted = False
            self._apply_update(-step_vec)  # undo
            self.lr = max(self.min_lr, self.dec_factor * self.lr)
            return self.state["prev_loss"]  # unchanged loss

        # 6. store state for next iteration
        self.state["prev_loss"] = new_loss_val
        self.state["prev_grad"] = _flat_grad(self.model).clone()
        self.state["prev_params"] = _flat_params(self.model)

        return new_loss_val
