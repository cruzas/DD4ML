from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer

from dd4ml.optimizers.lsr1 import LSR1
from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_tr_hparams, solve_tr_first_order, solve_tr_second_order


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
        self.tol = float(kwargs.pop("tol", 1e-6))
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
            mem_len = self.mem_length
            device = self.ps[0].device
            self.hess = LSR1(
                gamma=1.0, memory_length=mem_len, device=device, tol=self.tol
            )
            self.obs = OBS()
        else:
            self.hess = None  # type: ignore
            self.obs = None  # type: ignore

    def _flat_grad(self) -> torch.Tensor:
        """Return the current gradient as a single flat vector."""
        self._grad_buf.zero_()
        for i, p in enumerate(self.ps):
            if p.grad is not None:
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                g = p.grad
                if isinstance(g, WeightParallelizedTensor):
                    g = g.detach()
                self._grad_buf[s:e].copy_(g.view(-1))
        return self._grad_buf.clone()

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
        grad = _["grad"].clone() if "grad" in _ else self._flat_grad()
        if isinstance(grad, WeightParallelizedTensor):
            grad = grad.detach()
        gn = torch.norm(grad, p=self.norm_type)

        # Convergence test
        if gn <= self.tol:
            return loss, grad

        # First- or second-order TR step
        if self.second_order and self.hess and len(self.hess._S) > 0:
            self._step_buf, pred_red = solve_tr_second_order(
                grad,
                gn,
                self.delta,
                self.hess,  # type: ignore[arg-type]
                self.obs,  # type: ignore[arg-type]
                self.tol,
            )
        else:
            self._step_buf, pred_red = solve_tr_first_order(
                grad, gn, self.delta, self.tol
            )

        # Trial step
        self._apply_update()
        trial_loss = closure(compute_grad=True)
        trial_grad = self._flat_grad()

        # Acceptance ratio ρ
        if abs(float(pred_red)) < self.tol:
            rho = float("inf")  # Avoid division by zero
        else:
            rho = (loss - trial_loss) / pred_red

        if rho > self.nu_dec:
            # Accept
            if self.second_order:
                # Update Hessian memory
                sk = self._step_buf.clone()
                yk = (trial_grad - grad).clone()
                
                if sk.norm() > self.tol and yk.norm() > self.tol:
                    # Also takes care of updating gamma
                    self.hess.update_memory(sk, yk)

            if rho > self.nu_inc and self._step_buf.norm() >= 0.9 * self.delta:
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
