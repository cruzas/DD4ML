from __future__ import annotations
from typing import Iterable, Tuple
import torch
from torch.optim import Optimizer
from dd4ml.optimizers.lsr1 import LSR1
from dd4ml.solvers.obs import OBS
from dd4ml.utility import (
    get_tr_hparams,
    solve_tr_first_order,
    solve_tr_second_order,
)

class TR(Optimizer):
    __name__ = "TR"
    
    @staticmethod
    def setup_TR_hparams(cfg):
        # Add trust-region hyperparameters to the config
        for k, v in get_tr_hparams(cfg).items():
            setattr(cfg, k, v)
        return cfg

    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        # Extract trust-region hyperparameters from kwargs
        self.delta = kwargs.pop("delta", 0.1)
        self.norm_type = kwargs.pop("norm_type", 2)
        self.tol = kwargs.pop("tol", 1e-6)
        self.second_order = bool(kwargs.pop("second_order", False))
        self.mem_length = int(kwargs.pop("mem_length", 10))
        self.nu_dec = kwargs.pop("nu_dec")
        self.nu_inc = kwargs.pop("nu_inc")
        self.max_delta = kwargs.pop("max_delta")
        self.inc_factor = kwargs.pop("inc_factor")
        self.min_delta = kwargs.pop("min_delta")
        self.dec_factor = kwargs.pop("dec_factor")

        # Only 'lr' remains in defaults
        defaults = {"lr": self.delta}
        super().__init__(params, defaults)

        # Flatten parameter list
        self.ps = [p for g in self.param_groups for p in g["params"]]
        self.shapes = [p.shape for p in self.ps]
        self.numels = [p.numel() for p in self.ps]

        # Offsets into the flat buffers
        self.offsets = torch.tensor([0] + self.numels).cumsum(0)
        total = int(self.offsets[-1])

        # Reusable buffers
        device = self.ps[0].device
        self._grad_buf = torch.zeros(total, device=device)
        self._step_buf = torch.zeros_like(self._grad_buf)

        # Optional second-order support
        if self.second_order:
            tol = self.tol
            mem_len = self.mem_length
            device = self.ps[0].device
            self.hess = LSR1(1.0, mem_len, device, tol)
            self.obs = OBS()
        else:
            self.hess = None  # type: ignore
            self.obs = None   # type: ignore

    def _flat_grad(self) -> torch.Tensor:
        """Return the current gradient as a single flat vector."""
        self._grad_buf.zero_()
        for i, p in enumerate(self.ps):
            if p.grad is not None:
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                self._grad_buf[s:e].copy_(p.grad.view(-1))
        return self._grad_buf

    def _apply_update(self, sign: float = 1.0) -> None:
        """Add sign * step to each parameter tensor in-place."""
        with torch.no_grad():
            for i, p in enumerate(self.ps):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                p.add_(self._step_buf[s:e].view(self.shapes[i]) * sign)

    def update_pytorch_lr(self) -> None:
        """Keep PyTorch's recorded lr in sync with the current δ."""
        for g in self.param_groups:
            g["lr"] = self.delta

    def step(self, closure, **_) -> Tuple[float, torch.Tensor]:
        # Evaluate loss and gradient
        loss = _["loss"] if "loss" in _ else closure(compute_grad=True)
        grad = _["grad"] if "grad" in _ else self._flat_grad()
        gn = torch.norm(grad, p=self.norm_type)

        # Convergence test
        if gn <= self.tol:
            return loss, grad

        # First- or second-order TR step
        use_second = (
            self.second_order
            and self.hess
            and len(self.hess._S) > 0  # type: ignore
        )
        if use_second:
            self._step_buf, predicted = solve_tr_second_order(
                grad,
                gn,
                self.delta,
                self.hess,      # type: ignore[arg-type]
                self.obs,       # type: ignore[arg-type]
                self.tol,
            )
        else:
            self._step_buf, predicted = solve_tr_first_order(
                grad, gn, self.delta, self.tol
            )

        # Trial step
        self._apply_update()
        trial_loss = closure(compute_grad=True)
        trial_grad = self._flat_grad()

        # Acceptance ratio ρ
        actual = (loss - trial_loss)
        rho = actual / (predicted + 1e-12)

        if actual > 0 and rho >= self.nu_dec:
            # Accept
            if use_second:
                self.hess.update_memory(  # type: ignore[union-attr]
                    self._step_buf.clone(), (trial_grad - grad).clone()
                )
            # Optional expansion of trust region 
            if (
                rho >= self.nu_inc
                and self._step_buf.norm() >= 0.9 * self.delta
            ):
                self.delta = min(
                    self.max_delta,
                    self.inc_factor * self.delta,
                )
                self.update_pytorch_lr()
            return trial_loss, trial_grad

        # Reject
        self._apply_update(sign=-1.0)
        self.delta = max(
            self.min_delta,
            self.dec_factor * self.delta,
        )
        self.update_pytorch_lr()
        return loss, grad
