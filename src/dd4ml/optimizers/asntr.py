from __future__ import annotations

from typing import Callable, Iterable, List

import torch
from torch import Tensor, nn
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
    """Adaptive Sampled Newton-Trust Region (ASNTR) optimizer."""

    __name__ = "ASNTR"

    # ------------------------------------------------------------------ #
    # Top-level helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def setup_ASNTR_hparams(cfg):
        for k, v in get_asntr_hparams(cfg).items():
            setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
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
        norm_type: int = 2,
        c_1: float = 1.0,
        c_2: float = 100,
        alpha: float = 1.1,
        tol: float = 1e-8,
    ) -> None:
        defaults = {"lr": lr}  # kept for API compatibility
        super().__init__(params, defaults)

        self.device = torch.device(device)

        # Trust-region radii
        self.delta = float(delta)
        self.max_delta = float(max_delta)
        self.tol = float(tol)
        self.second_order = bool(second_order)

        # SR1 memory and OBS solver for the TR sub-problem
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
        self.c_1 = c_1
        self.c_2 = c_2
        self.alpha = alpha  # α > 1 for decreasing tₖ and t̃ₖ
        self.dataset_len = 0  # to be initialized by the user (N in the paper)

        # Iteration counter
        self.k = 0

        # ------------------------------------------------------------------
        # Optimizer state
        # ------------------------------------------------------------------
        # Store previous step and gradient in the *global* state dictionary so
        # that they are included in ``state_dict`` serialisation, akin to
        # LSSR1_TR.
        st = self.state
        st["prev_s"] = None  # type: Tensor | None
        st["prev_g"] = None  # type: Tensor | None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _all_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                yield p

    def _flatgrad(self) -> Tensor:
        pieces: List[Tensor] = []
        for p in self._all_params():
            if p.grad is None:
                pieces.append(torch.zeros_like(p).view(-1))
            else:
                g = p.grad
                if isinstance(g, WeightParallelizedTensor):
                    g = g.detach()
                pieces.append(g.view(-1))
        return torch.cat(pieces)

    def _apply_flat(self, step: Tensor, sign: float) -> None:
        idx = 0
        with torch.no_grad():
            for p in self._all_params():
                numel = p.numel()
                p.data.add_(sign * step[idx : idx + numel].view_as(p))
                idx += numel

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(
        self,
        *,
        closure_main: Callable[[bool], Tensor],
        closure_d: Callable[[bool], Tensor],
        hNk=None,
        **_,
    ) -> float:
        """
        Perform one ASNTR step.

        The method follows the algorithmic structure from the original
        implementation but now interacts with ``self.state`` instead of the
        removed ``self._prev_s`` and ``self._prev_g`` attributes.
        """

        st = self.state

        # Trust-region penalties
        tk = self.c_1 / (self.k + 1) ** self.alpha
        ttilde_k = self.c_2 / (self.k + 1) ** self.alpha

        fN_old = _["loss"] if "loss" in _ else closure_main(compute_grad=True)
        g = _["grad"] if "grad" in _ else self._flatgrad().detach()

        fD_old = closure_d(compute_grad=True)  # grads for control batch
        g_bar = self._flatgrad().detach()

        if st["prev_s"] is not None:
            self.hess.update_memory(st["prev_s"], g - st["prev_g"])  # type: ignore[arg-type]

        gn = torch.norm(g, p=self.norm_type)
        if self.second_order and len(self.hess._S) > 0:
            step, pred_red = solve_tr_second_order(
                g, gn, self.delta, self.hess, self.obs, self.tol
            )
        else:
            step, pred_red = solve_tr_first_order(g, gn, self.delta, self.tol)

        self._apply_flat(step, +1)  # move to trial point
        with torch.no_grad():
            fN_new = closure_main(compute_grad=False)
            fD_new = closure_d(compute_grad=False)

        rho_N = (fN_old - fN_new + tk * self.delta) / (pred_red + 1e-12)
        rho_D = (fD_old - fD_new + ttilde_k * self.delta) / (-g_bar.dot(step) + 1e-12)

        accepted = rho_N >= self.eta and rho_D >= self.nu
        if accepted:
            st["prev_s"] = step.clone().detach()
            st["prev_g"] = g.clone().detach()
        else:  # reject - roll back
            self._apply_flat(step, -1)
            st["prev_s"] = None
            st["prev_g"] = None

        if rho_N < self.eta_1:
            self.delta *= self.tau_1
        elif (
            rho_N > self.eta_2
            and torch.norm(step, p=self.norm_type) > self.tau_2 * self.delta
        ):
            self.delta = min(self.delta * self.tau_3, self.max_delta)

        # Clip trust-region radius between [min_delta, max_delta]
        self.delta = max(
            self.min_delta,
            min(self.delta, self.max_delta),
        )

        # Increment iteration counter
        self.k += 1
        return fN_new
