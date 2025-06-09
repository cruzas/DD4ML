from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from dd4ml.optimizers.lsr1 import LSR1
from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.solvers.obs import OBS
from dd4ml.utility.optimizer_utils import (
    get_asntr_hparams,
    solve_tr_first_order,
    solve_tr_second_order,
)


class ASNTR(Optimizer):
    """Adaptive Sampled Newton Trust Region optimizer."""

    __name__ = "ASNTR"

    @staticmethod
    def setup_ASNTR_hparams(cfg):
        for k, v in get_asntr_hparams(cfg).items():
            setattr(cfg, k, v)
        return cfg

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device | str = "cpu",
        lr: float = 1.0,
        delta: float = 1.0,
        max_delta: float = 10.0,
        gamma: float = 1e-3,
        second_order: bool = True,
        mem_length: int = 30,
        eta: float = 1e-4,
        nu: float = 1e-4,
        eta_1: float = 0.1,
        eta_2: float = 0.75,
        tau_1: float = 0.5,
        tau_2: float = 0.8,
        tau_3: float = 2.0,
        C_1: float = 1.0,
        C_2: float = 1.0,
        alpha: float = 1.1,
        tol: float = 1e-8,
    ) -> None:
        defaults = {"lr": delta}
        super().__init__(params, defaults)

        self.model = model
        self.criterion = criterion
        self.device = torch.device(device)

        self.delta = float(delta)
        self.max_delta = float(max_delta)
        self.tol = float(tol)
        self.second_order = bool(second_order)

        # SR1 memory and OBS solver for TR sub-problem
        self.hess = LSR1(gamma=gamma, memory_length=mem_length, device=self.device)
        self.obs = OBS()

        # Algorithmic constants
        self.eta = eta
        self.nu = nu
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.C_1 = C_1
        self.C_2 = C_2
        self.alpha = alpha

        # State
        self.k = 0
        self._prev_s: Tensor | None = None
        self._prev_g: Tensor | None = None

    # ------------------------------------------------------------------
    def _flat_params(self) -> List[Tensor]:
        return [p for p in self.model.parameters()]

    def _flatgrad(self) -> Tensor:
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p).view(-1))
            else:
                g = p.grad
                if isinstance(g, WeightParallelizedTensor):
                    g = g.detach()
                grads.append(g.view(-1))
        return torch.cat(grads)

    def _apply_flat(self, step: Tensor, sign: float) -> None:
        idx = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p.data.add_(sign * step[idx : idx + numel].view_as(p))
                idx += numel

    # ------------------------------------------------------------------
    def step(
        self,
        *,
        inputs: Tensor,
        labels: Tensor,
        inputs_d: Tensor,
        labels_d: Tensor,
    ) -> float:
        """Perform one ASNTR step using two batches of data."""

        tk = self.C_1 / (self.k + 1) ** self.alpha
        ttilde_k = self.C_2 / (self.k + 1) ** self.alpha

        self.model.train()
        for p in self.model.parameters():
            p.grad = None

        # Loss and gradient on main batch
        fN_old = self.criterion(self.model(inputs), labels)
        fN_old.backward()
        g = self._flatgrad().detach()

        # Control batch
        for p in self.model.parameters():
            p.grad = None
        fD_old = self.criterion(self.model(inputs_d), labels_d)
        fD_old.backward()
        g_bar = self._flatgrad().detach()

        for p in self.model.parameters():
            p.grad = None

        # Update SR1 memory
        if self._prev_s is not None:
            self.hess.update_memory(self._prev_s, g - self._prev_g)  # type: ignore[arg-type]

        gn = g.norm()
        if self.second_order and len(self.hess._S) > 0:
            step, pred_red = solve_tr_second_order(
                g,
                gn,
                self.delta,
                self.hess,
                self.obs,
                self.tol,
            )
        else:
            step, pred_red = solve_tr_first_order(g, gn, self.delta, self.tol)

        # Trial evaluation
        self._apply_flat(step, +1)
        with torch.no_grad():
            fN_new = self.criterion(self.model(inputs), labels)
            fD_new = self.criterion(self.model(inputs_d), labels_d)

        rho_N = (fN_old - fN_new + tk * self.delta) / (pred_red + 1e-12)
        rho_D = (fD_old - fD_new + ttilde_k * self.delta) / (-g_bar.dot(step) + 1e-12)

        accepted = rho_N >= self.eta and rho_D >= self.nu
        if accepted:
            self._prev_s = step.clone().detach()
            self._prev_g = g.clone().detach()
        else:
            self._apply_flat(step, -1)
            self._prev_s = None
            self._prev_g = None

        if rho_N < self.eta_1:
            self.delta *= self.tau_1
        elif rho_N > self.eta_2 and step.norm() > self.tau_2 * self.delta:
            self.delta = min(self.delta * self.tau_3, self.max_delta)

        self.k += 1

        return float(fN_new.detach())
