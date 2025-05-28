"""lssr1.py - Limited-Memory Stochastic SR1 Trust-Region Optimiser
------------------------------------------------------------------
ASCII-only implementation of **LSSR1_TR** that delegates

* Hessian modelling to   ``hessian_approx.LSR1``
* SR1 trust-region solves to ``obs.OBS``

It follows the radius-update rule of Algorithm 3 in the paper and now
includes two explicit stopping criteria:

1. Optimisation is declared converged when ``||g_k|| <= tol``.
2. The ``step`` method returns the tuple ``(loss, grad_norm)``, where
   ``grad_norm`` is the Euclidean norm of the gradient *at the parameters*
   that are finally accepted (original or updated).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple

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
# PyTorch optimizer implementing Algorithm 3 (L-SSR1-TR)
# -----------------------------------------------------------------------------


class LSSR1_TR(Optimizer):
    r"""Limited-Memory **Stochastic SR1** Trust-Region optimizer.

    The trust-region radius ``delta`` is updated exactly as in Algorithm 3
    with user-exposed parameters::

        0 <= tau_1 < tau_2 < 0.5 < tau_3 < 1
        0 <  nu_1 < nu_2 <= 0.5 < nu_3 < 1 < nu_4

    Convergence is declared when the Euclidean norm of the full gradient
    is below ``tol``.

    The :meth:`step` routine returns ``(loss, grad_norm)`` where
    ``grad_norm`` corresponds to the parameters that are actually retained
    (i.e., after the line-search decision).
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        params,
        *,
        lr_init: float = 1.0,
        delta_init: float = 1.0,
        gamma_init: float = 1e-3,
        mem_length: int = 10,
        mu: float = 0.9,
        tau_1: float = 0.1,
        tau_2: float = 0.25,
        tau_3: float = 0.75,
        nu_1: float = 0.25,
        nu_2: float = 0.5,
        nu_3: float = 0.8,
        nu_4: float = 2.0,
        tol: float = 1e-8,
    ):
        if not (0.0 <= tau_1 < tau_2 < 0.5 < tau_3 < 1.0):
            raise ValueError(
                "tau parameters must satisfy 0 <= tau_1 < tau_2 < 0.5 < tau_3 < 1."
            )
        if not (0.0 < nu_1 < nu_2 <= 0.5 < nu_3 < 1.0 < nu_4):
            raise ValueError(
                "nu parameters must satisfy 0 < nu_1 < nu_2 <= 0.5 < nu_3 < 1 < nu_4."
            )

        defaults = dict(
            lr_init=lr_init,
            delta=delta_init,
            gamma=gamma_init,
            mem_length=mem_length,
            mu=mu,
            tol=tol,
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
            memory_length=mem_length,
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

    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[Tensor, float]:
        """Perform one optimisation step.

        Returns
        -------
        (loss, grad_norm)
            * loss - objective value before any parameter update
            * grad_norm - ||∇f||₂ at the parameters finally accepted
        """
        # -- evaluate loss and gradient at current parameters ------------
        params = self.param_groups[0]["params"]

        if "precomp_loss" in _:
            loss = float(_["precomp_loss"])
        else:
            loss = closure()  # forward & backward pass
        if "precomp_grad" in _
            g = _["precomp_grad"]
        else:
            g = _concat_grads(params)

        g_norm = float(g.norm())
        # -- convergence test --------------------------------------------
        if g_norm <= self.defaults["tol"]:
            return loss, g

        # -- current point bookkeeping -----------------------------------
        wk = _concat_params(params)
        st = self.state
        if st["wk"] is None:
            st["wk"] = wk.clone()
        if st["vk"].numel() != wk.numel():
            st["vk"] = torch.zeros_like(wk)

        # ----------------------------------------------------------------
        # 1. Update limited-memory pairs
        # ----------------------------------------------------------------
        if st["prev_grad"] is not None:
            s_vec = wk - st["wk"]
            y_vec = g - st["prev_grad"]
            self.hess.update_memory(s_vec, y_vec)
        st["wk"], st["prev_grad"] = wk.clone(), g.clone()
        self.hess.precompute()

        # ----------------------------------------------------------------
        # 2. Solve trust-region sub-problem via OBS
        # ----------------------------------------------------------------
        p_star = self.obs.solve_tr_subproblem(
            g,
            delta=self.defaults["delta"],
            gamma=self.hess.gamma,
            Psi=self.hess.Psi,
            Minv=self.hess.M_inv,
        )

        # ----------------------------------------------------------------
        # 3. Momentum grafting (Eq. 17)
        # ----------------------------------------------------------------
        vk_prev = st["vk"]
        vk = vk_prev * self.defaults["mu"] + (wk - st["wk"])  # previous step
        if vk.norm() > 0:
            vk = self.defaults["mu"] * min(1.0, self.defaults["delta"] / vk.norm()) * vk
        p_comb = p_star + vk
        if p_comb.norm() > 0:
            p_comb = min(1.0, self.defaults["delta"] / p_comb.norm()) * p_comb
        st["vk"] = vk.clone()

        # ----------------------------------------------------------------
        # 4. Wolfe back-tracking line-search along p_comb
        # ----------------------------------------------------------------
        alpha = self.defaults["lr_init"]
        c1 = 1e-4
        orig_loss = loss.item()
        grad_dot_dir = g.dot(p_comb)

        for wls in range(10):
            _set_param_vector(params, wk + alpha * p_comb)
            new_loss = closure().item()  # new gradients
            if new_loss <= orig_loss + c1 * alpha * grad_dot_dir:  # Armijo
                break
            alpha *= 0.5
        else:
            alpha = 0.0  # reject step
            _set_param_vector(params, wk)

        p_alpha = alpha * p_comb  # accepted step

        # ----------------------------------------------------------------
        # 5. Trust-region radius update (Algorithm 3)
        # ----------------------------------------------------------------
        delta_old = self.defaults["delta"]
        rho = 0.0
        if alpha > 0:  # predicted reduction
            pred = g.dot(p_alpha)  # linear term
            if pred < 0:
                rho = (new_loss - orig_loss) / pred

        s_norm = p_alpha.norm()

        if rho < self.defaults["tau_2"]:
            delta_new = min(
                self.defaults["nu_1"] * delta_old,
                self.defaults["nu_2"] * (s_norm**2),
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

        # ----------------------------------------------------------------
        # 6. Gradient-norm selection for return value
        # ----------------------------------------------------------------
        if alpha == 0.0:
            grad_norm = g_norm  # kept old point
        else:
            grad_norm = float(_concat_grads(params).norm())  # accepted new point

        return loss, g
