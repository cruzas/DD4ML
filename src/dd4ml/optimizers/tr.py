from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from dd4ml.optimizers.hessian_approx import LSR1
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_trust_region_params


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #
def _flat_grad(params) -> torch.Tensor:
    """Return concatenated, detached gradients of *params*."""
    gs: List[torch.Tensor] = [
        p.grad.detach().flatten() for p in params if p.grad is not None
    ]
    return torch.cat(gs) if gs else torch.zeros(0, dtype=torch.float32)


# --------------------------------------------------------------------- #
# optimiser                                                             #
# --------------------------------------------------------------------- #
class TR(Optimizer):
    # Default values for optimizer hyperparameters
    defaults = dict(
        delta=0.01,
        max_delta=1.0,
        min_delta=1e-4,
        nu=0.5,
        inc_factor=2.0,
        dec_factor=0.5,
        nu_dec=0.25,
        nu_inc=0.75,
        mem_length=5,
        norm_type=2,
        second_order=True,
        tol=1e-6,
    )

    @staticmethod
    def setup_TR_args(cfg):
        for k, v in get_trust_region_params(cfg).items():
            setattr(cfg, k, v)
        return cfg

    # ---------------- constructor ------------------------------------- #
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        delta: float = 0.01,
        max_delta: float = 1.0,
        min_delta: float = 1e-4,
        nu: float = 0.5,
        inc_factor: float = 2.0,
        dec_factor: float = 0.5,
        nu_dec: float = 0.25,
        nu_inc: float = 0.75,
        mem_length: Optional[int] = 5,
        norm_type: int = 2,
        *,
        second_order: bool = True,
        tol: float = 1e-6,
    ) -> None:
        # Use defaults for param_groups
        defaults = dict(
            delta=delta,
            max_delta=max_delta,
            min_delta=min_delta,
            nu=nu,
            inc_factor=inc_factor,
            dec_factor=dec_factor,
            nu_dec=nu_dec,
            nu_inc=nu_inc,
            mem_length=mem_length,
            norm_type=norm_type,
            second_order=second_order,
            tol=tol,
            lr=delta,  # PyTorch expects 'lr'
        )
        super().__init__(params, defaults)

        self.ps = [p for g in self.param_groups for p in g["params"]]

        # trust-region radii / factors
        group = self.param_groups[0]
        self.delta = float(group.get("delta", self.defaults["delta"]))
        self.max_delta = float(group.get("max_delta", self.defaults["max_delta"]))
        self.min_delta = float(group.get("min_delta", self.defaults["min_delta"]))
        self.inc_factor = float(group.get("inc_factor", self.defaults["inc_factor"]))
        self.dec_factor = float(group.get("dec_factor", self.defaults["dec_factor"]))
        self.nu_dec = float(group.get("nu_dec", self.defaults["nu_dec"]))
        self.nu_inc = float(group.get("nu_inc", self.defaults["nu_inc"]))
        self.nu = min(group.get("nu", self.defaults["nu"]), self.nu_dec)
        self.norm_type = group.get("norm_type", self.defaults["norm_type"])
        self.tol = float(group.get("tol", self.defaults["tol"]))

        # second-order helpers
        self.second_order = bool(group.get("second_order", self.defaults["second_order"]))
        if self.second_order:
            dev = self.ps[0].device
            self.hess = LSR1(
                gamma=1.0,
                memory_length=int(group.get("mem_length", self.defaults["mem_length"]) or 5),
                device=dev,
                tol=1e-10,
            )
            self.obs = OBS()
        else:
            self.hess = None  # type: ignore
            self.obs = None  # type: ignore

    # ---------------- utilities --------------------------------------- #
    def _apply_update(self, step: torch.Tensor) -> None:
        """Add flattened *step* to parameters (no grad tracking)."""
        offset = 0
        with torch.no_grad():
            for p in self.ps:
                num = p.numel()
                p.add_(step[offset : offset + num].view_as(p))
                offset += num

    # ---------------- sub-problem solvers ----------------------------- #
    def _solve_tr_first_order(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        g_norm = torch.norm(g, p=self.norm_type)
        if g_norm == 0:
            return torch.zeros_like(g), 0.0
        step = -g * (delta / g_norm)
        predicted = delta * g_norm  # since B ≈ I
        return step, float(predicted)

    def _solve_tr_second_order(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        self.hess.precompute()  # type: ignore
        step = -self.obs.solve_tr_subproblem(  # type: ignore
            g,
            torch.tensor(delta, device=g.device, dtype=g.dtype),
            self.hess.gamma,
            self.hess.Psi,
            self.hess.Minv,  # type: ignore
        )
        g_dot_p = torch.dot(g, step)
        p_B_p = torch.dot(step, self.hess.B(step))  # type: ignore
        predicted = -(g_dot_p + 0.5 * p_B_p)
        return step, float(predicted)

    def update_pytorch_lr(self) -> None:
        """Update the learning rate in PyTorch's param_groups."""
        for g in self.param_groups:
            g["lr"] = self.delta

    # -------------------------- main loop ----------------------------- #
    def step(self, closure, **_) -> Tuple[float, torch.Tensor]:
        """Execute one trust-region update.
        Returns (loss_value, accepted_gradient)."""
        # current objective and gradient
        loss_val = (
            float(_["precomp_loss"])
            if "precomp_loss" in _
            else float(closure(compute_grad=True))
        )
        grad = _["precomp_grad"] if "precomp_grad" in _ else _flat_grad(self.ps)

        # convergence check
        if torch.norm(grad, p=self.norm_type) <= self.tol:
            return loss_val, grad

        # select sub-problem solver
        solve = (
            self._solve_tr_second_order
            if self.second_order and len(self.hess._S) > 0  # type: ignore[attr-defined]
            else self._solve_tr_first_order
        )
        step, predicted = solve(grad, self.delta)

        # trial step
        self._apply_update(step)
        new_loss = float(closure(compute_grad=True))
        new_grad = _flat_grad(self.ps)

        # acceptance test
        actual = loss_val - new_loss
        rho = actual / (predicted + 1e-12)

        if actual > 0 and rho >= self.nu_dec:  # accepted
            if rho >= self.nu_inc and torch.norm(step) >= 0.9 * self.delta:
                self.delta = min(self.max_delta, self.inc_factor * self.delta)
                self.update_pytorch_lr()

            if self.second_order:
                self.hess.update_memory(
                    step.clone(), (new_grad - grad).clone()
                )  # type: ignore
            return new_loss, new_grad
        else:  # rejected
            self._apply_update(-step)  # rollback
            self.delta = max(self.min_delta, self.dec_factor * self.delta)
            self.update_pytorch_lr()
            return loss_val, grad
