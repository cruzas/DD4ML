import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer

from dd4ml.solvers.obs import OBS  # closed‑form TR sub‑solver
from dd4ml.utility import get_trust_region_params

from .hessian_approx import HessianApproxSR1  # compact L‑SR1 factors


class TrustRegionSecondOrder(Optimizer):
    """Stochastic limited-memory SR1 trust-region optimiser using the
    OBS closed-form algorithm (Fletcher & Gould, 2021) for the inner
    sub-problem.

    The class follows the notation in Nocedal & Wright (2006):
      minimise m(p) = g^T p + ½ p^T B p  subject to ‖p‖ ≤ Δ.

    B is the limited-memory SR1 Hessian approximation stored in compact
    form (gamma, Ψ, M⁻¹).  OBS provides the exact global minimiser of the
    sub-problem; the outer loop accepts/rejects using the usual ratio
    rjp = (f(x)-f(x+p)) / (m(0)-m(p)).
    """

    # ------------------------------------------------------------------
    # Convenience helper for hydra / argparse configs ------------------
    # ------------------------------------------------------------------
    @staticmethod
    def setup_TR_args(config):
        params = get_trust_region_params(config)
        config.max_iter = params["max_iter"]
        config.lr = params["lr"]
        config.max_lr = params["max_lr"]
        config.min_lr = params["min_lr"]
        config.nu = params["nu"]
        config.inc_factor = params["inc_factor"]
        config.dec_factor = params["dec_factor"]
        config.nu_1 = params["nu_1"]
        config.nu_2 = params["nu_2"]
        config.norm_type = params["norm_type"]
        return config

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.01,
        max_lr: float = 1.0,
        min_lr: float = 1e-4,
        nu: float = 0.5,
        inc_factor: float = 2.0,
        dec_factor: float = 0.5,
        nu_1: float = 0.25,
        nu_2: float = 0.75,
        max_iter: int = 10,
        norm_type: int = 2,
        memory_length: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model.parameters(), {"lr": lr})

        self.model = model
        self.param_list = list(model.parameters())
        self.device = device or next(model.parameters()).device

        # trust‑region radii
        self.lr = float(lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.inc_factor = float(inc_factor)
        self.dec_factor = float(dec_factor)
        self.nu_1 = float(nu_1)
        self.nu_2 = float(nu_2)
        self.nu = min(nu, self.nu_1)
        self.norm_type = norm_type

        # limited‑memory SR1 components & OBS solver
        self.hess = HessianApproxSR1(memory_length=memory_length, device=self.device)
        self.obs = OBS()

        # bookkeeping for curvature pairs
        self.prev_loss: Optional[float] = None
        self.prev_grad: Optional[torch.Tensor] = None  # flat
        self.prev_params: Optional[torch.Tensor] = None  # flat
        self.initialized: bool = False

    # ------------------------------------------------------------------
    # Helpers -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _gather_flat_grad(self) -> torch.Tensor:
        grads = [p.grad.view(-1) for p in self.param_list if p.grad is not None]
        return torch.cat(grads) if grads else torch.tensor([], device=self.device)

    def _apply_update(self, step: torch.Tensor) -> None:
        """Add *step* (flat) to the parameters inplace."""
        offset = 0
        for p in self.param_list:
            n = p.numel()
            if n == 0:
                continue
            p.data.add_(step[offset : offset + n].view_as(p.data))
            offset += n

    # ------------------------------------------------------------------
    # TR sub‑problem via OBS -------------------------------------------
    # ------------------------------------------------------------------
    def _solve_tr_subproblem_obs(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        """Return p* and predicted reduction m(0)−m(p*)."""
        # ensure compact factors are up‑to‑date
        self.hess.precompute()

        # OBS expects tensors for all arguments
        p = -self.obs.solve_tr_subproblem(
            g,
            torch.tensor(delta, device=g.device, dtype=g.dtype),
            self.hess.gamma,
            self.hess.Psi,
            self.hess.M_inv,
        )

        # quadratic model decrease  m(p) = g^T p + ½ p^T B p
        gTp = torch.dot(g, p)
        Bp = self.hess.B(p)
        predicted = -(gTp + 0.5 * torch.dot(p, Bp))
        return p, float(predicted)

    # ------------------------------------------------------------------
    # Main optimisation step -------------------------------------------
    # ------------------------------------------------------------------
    def step(self, closure, compute_grad: bool = True):
        """Perform one trust-region iteration.

        *closure* must accept keyword *compute_grad* and return the loss.
        """
        # ------------------------------------------------------------------
        # evaluate f(x) and g(x)
        # ------------------------------------------------------------------
        loss_tensor = closure(compute_grad=compute_grad)
        loss_val = float(loss_tensor.item())
        g = self._gather_flat_grad()
        if g.numel() == 0:
            return loss_val  # nothing to do

        # ------------------------------------------------------------------
        # initialise memory on first call
        # ------------------------------------------------------------------
        if not self.initialized:
            params_flat = torch.cat([p.data.view(-1) for p in self.param_list])
            self.prev_params = params_flat.clone()
            self.prev_grad = g.clone()
            self.prev_loss = loss_val
            self.initialized = True

        # ------------------------------------------------------------------
        # solve TR sub‑problem (OBS)
        # ------------------------------------------------------------------
        p, pred_red = self._solve_tr_subproblem_obs(g, self.lr)

        # save copy of current parameters for possible rollback
        current_params = torch.cat([p_.data.view(-1) for p_ in self.param_list])

        # tentative step
        self._apply_update(p)

        # evaluate new loss and gradient at x + p
        new_loss_tensor = closure(compute_grad=compute_grad)
        new_loss = float(new_loss_tensor.item())
        new_grad = self._gather_flat_grad()

        # ------------------------------------------------------------------
        # acceptance test
        # ------------------------------------------------------------------
        act_red = self.prev_loss - new_loss
        rho = act_red / (pred_red + 1e-12)

        if act_red > 0 and rho >= self.nu_1:
            # accept -----------------------------------------------------
            if rho >= self.nu_2 and torch.norm(p, p=self.norm_type) >= 0.9 * self.lr:
                self.lr = min(self.max_lr, self.lr * self.inc_factor)
            # curvature pair (s,y)
            s_vec = (
                torch.cat([p_.data.view(-1) for p_ in self.param_list])
                - self.prev_params
            )
            y_vec = new_grad - self.prev_grad
            self.hess.update_memory(s_vec, y_vec)

            # update reference point
            self.prev_params = torch.cat([p_.data.view(-1) for p_ in self.param_list])
            self.prev_grad = new_grad.clone()
            self.prev_loss = new_loss
            return new_loss
        else:
            # reject -----------------------------------------------------
            # rollback parameters and shrink radius
            offset = 0
            for param in self.param_list:
                n = param.numel()
                if n == 0:
                    continue
                param.data.copy_(current_params[offset : offset + n].view_as(param))
                offset += n
            self.lr = max(self.min_lr, self.lr * self.dec_factor)
            return self.prev_loss
