from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer

from dd4ml.optimizers.lsr1 import LSR1
from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_asntr_hparams, solve_tr_first_order, solve_tr_second_order


class ASNTR(Optimizer):
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
        device: str,
        lr: float = 1.0,
        delta: float = 1.0,
        max_delta: float = 10.0,
        gamma: float = 1e-3,
        second_order: bool = True,
        mem_length: int = 30,
        eta: float = 1e-4,  # 0 < eta < eta_2 <= 3/4
        nu: float = 1e-4,  # in (0, 1/4)
        eta_1: float = 0.1,  # in (eta, eta_2)
        eta_2: float = 0.75,
        tau_1: float = 0.5,  # 0 < tau_1 <= 0.5 < tau_2
        tau_2: float = 0.8,  # 0.5 < tau_2 < 1 < tau_3
        tau_3: float = 2.0,
        C_1: float = 1.0,
        C_2: float = 1.0,
        alpha: float = 1.1,
    ):
        # Network & data -------------------------------------------------
        self.model = model
        self.device = device
        self.crit = nn.CrossEntropyLoss()
        self.imgs = train_imgs  # kept on CPU
        self.lbls = train_lbls
        self.N_total = len(train_lbls)

        # Adaptive sample size (Alg. 1, l.1)
        self.N0 = 128  # user can tune – here fixed for CIFAR10
        self.Nk = self.N0

        # Limited-memory SR1 storage
        self.mem = _SR1Mem(m=m, gamma0=lr)

        # Trust-region radii
        self.delta = delta
        self.max_delta = max_delta

        # Save hyper-parameters with paper names
        self.eta = eta
        self.nu = nu
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.C_1, self.C_2, self.alpha = C_1, C_2, alpha

    def _flat_params(self) -> List[Tensor]:
        return [p for p in self.model.parameters()]

    def _flatgrad(self) -> Tensor:
        return torch.cat([p.grad.flatten() for p in self._flat_params()])

    def _apply_flat(self, step: Tensor, sign: float) -> None:
        idx = 0
        for p in self._flat_params():
            numel = p.numel()
            p.data.add_(sign * step[idx : idx + numel].view_as(p))
            idx += numel

    # --------------------------------------------------------------------
    # Main optimisation step
    # --------------------------------------------------------------------
    def step(self) -> dict:
        # ------------------------- Sample-size rule (paper §4) ---------
        self.Nk = min(
            self.N_total,
            max(100 * self.k + self.N0, math.ceil(1 / (self.delta**2))),
        )

        tk = self.C_1 / (self.k + 1) ** self.alpha  # eq. (7)
        ttilde_k = self.C_2 / (self.k + 1) ** self.alpha  # eq. (10)

        # ------------------------- Draw subsamples ---------------------
        idx_N = torch.randperm(self.N_total)[: self.Nk]
        idx_D = torch.randint(0, self.N_total, (1,))

        x_N, y_N = self.imgs[idx_N].to(self.device), self.lbls[idx_N].to(self.device)
        x_D, y_D = self.imgs[idx_D].to(self.device), self.lbls[idx_D].to(self.device)

        # Normalisation (dataset specific – placeholders)
        x_N = x_N  # assume already normalised
        x_D = x_D

        # ------------------------- f_N(w_k), g_k -----------------------
        self.model.train()
        for p in self.model.parameters():
            p.grad = None
        fN_old = self.crit(self.model(x_N), y_N)
        fN_old.backward()
        g = self._flatgrad().detach()

        # ------------------------- Control sample (D_k) ----------------
        for p in self.model.parameters():
            p.grad = None
        fD_old = self.crit(self.model(x_D), y_D)
        fD_old.backward()
        g_bar = self._flatgrad().detach()
        for p in self.model.parameters():
            p.grad = None

        # ------------------------- SR1 memory update -------------------
        if self._prev_s is not None:
            self.mem.update(self._prev_s, g - self._prev_g)

        # ------------------------- Solve TR sub-problem ----------------
        p = _obs_solve(g, self.mem.apply_B, self.delta)
        pred_red = -(g.dot(p) + 0.5 * p.dot(self.mem.apply_B(p)))
        if pred_red <= 0:  # safeguard: take Cauchy step
            p = -g / g.norm() * self.delta
            pred_red = -(g.dot(p) + 0.5 * p.dot(self.mem.apply_B(p)))

        # ------------------------- Evaluate trial iterate --------------
        self._apply_flat(p, +1)
        with torch.no_grad():
            fN_new = self.crit(self.model(x_N), y_N)
            fD_new = self.crit(self.model(x_D), y_D)

        rho_N = (fN_old - fN_new + tk * self.delta) / (pred_red + 1e-12)
        rho_D = (fD_old - fD_new + ttilde_k * self.delta) / (-g_bar.dot(p) + 1e-12)

        # ------------------------- Acceptance test (Alg. 1 l.23-30) ----
        accepted = rho_N >= self.eta and (self.Nk == self.N_total or rho_D >= self.nu)
        if accepted:
            self._prev_s, self._prev_g = p.clone().detach(), g.clone().detach()
        else:
            self._apply_flat(p, -1)  # reject trial point
            self._prev_s = self._prev_g = None

        # ------------------------- TR radius update (Alg. 1 l.36-42) ---
        if rho_N < self.eta_1:
            self.delta *= self.tau_1
        elif rho_N > self.eta_2 and p.norm() > self.tau_2 * self.delta:
            self.delta = min(self.delta * self.tau_3, self.max_delta)
        # else: keep δ_k unchanged

        # ------------------------- Sample-size growth rule -------------
        if not accepted and self.Nk < self.N_total:
            self.Nk = min(self.N_total, math.ceil(1.01 * self.Nk))

        # Advance iteration counter
        self.k += 1

        # ------------------------- Diagnostics -------------------------
        return {
            "fN_old": fN_old.item(),
            "fN_new": fN_new.item(),
            "fD_old": fD_old.item(),
            "fD_new": fD_new.item(),
            "delta": self.delta,
            "Nk": self.Nk,
            "rho_N": rho_N.item(),
            "rho_D": rho_D.item(),
            "accepted": accepted,
        }
