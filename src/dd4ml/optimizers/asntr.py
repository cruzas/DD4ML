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

    @staticmethod
    def setup_ASNTR_hparams(cfg):
        for k, v in get_asntr_hparams(cfg).items():
            setattr(cfg, k, v)
        return cfg

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        device: torch.device | str = "cpu",
        lr: float = 1.0,
        delta: float = 1.0,
        min_delta: float = 1e-3,
        max_delta: float = 10.0,
        gamma: float = 1e-3,
        second_order: bool = True,
        mem_length: int = 30,
        # controls
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
        # flat buffer hooks
        flat_grads_fn: Callable[[], Tensor] | None = None,
        flat_params_fn: Callable[[], Tensor] | None = None,
        flat_params: WeightParallelizedTensor | None = None,
    ) -> None:
        defaults = {"lr": lr}
        super().__init__(params, defaults)

        self.device = torch.device(device)
        self.delta = float(delta)
        self.min_delta = float(min_delta)
        self.max_delta = float(max_delta)
        self.tol = float(tol)
        self.second_order = bool(second_order)

        # SR1 memory and OBS solver
        self.hess = LSR1(gamma=gamma, memory_length=mem_length, device=self.device)
        self.obs = OBS()

        # algorithmic constants
        self.eta = eta
        self.nu = nu
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.norm_type = norm_type
        self.c_1 = c_1
        self.c_2 = c_2
        self.alpha = alpha
        self.k = 0

        # precompute shapes and offsets for flatten/unflatten
        shapes: List[torch.Size] = []
        offsets = [0]
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            shapes.append(p.shape)
            offsets.append(offsets[-1] + n)
        total_size = offsets[-1]
        self._shapes = shapes
        self._offsets = offsets

        st = self.state
        # allocate flat buffers
        if flat_params is not None:
            st["flat_wk"] = flat_params.clone()
            st["flat_gk"] = flat_params.clone()
        else:
            buf = torch.zeros(total_size, device=self.device)
            st["flat_wk"] = buf
            st["flat_gk"] = buf.clone()

        # hooks
        self._flat_params_fn = (
            flat_params_fn if flat_params_fn is not None else self._flatten_params
        )
        self._flat_grads_fn = (
            flat_grads_fn if flat_grads_fn is not None else self._flatten_grads
        )

        # state for previous step
        st["prev_s"] = None
        st["prev_g"] = None

        # for external
        self.inc_batch_size = False
        self.move_to_next_batch = True

    def _flatten_params(self) -> Tensor:
        buf = self.state["flat_wk"]
        for p, start, end in zip(
            self.param_groups[0]["params"], self._offsets, self._offsets[1:]
        ):
            buf[start:end].copy_(p.data.view(-1))
        return buf

    def _flatten_grads(self) -> Tensor:
        buf = self.state["flat_gk"]
        for p, start, end in zip(
            self.param_groups[0]["params"], self._offsets, self._offsets[1:]
        ):
            buf[start:end].copy_(p.grad.view(-1))
        return buf

    def _unflatten_update(self, vec: Tensor) -> None:
        with torch.no_grad():
            if isinstance(vec, WeightParallelizedTensor):
                for p, shard in zip(self.param_groups[0]["params"], vec.tensor):
                    p.data.copy_(shard.view_as(p))
            else:
                for p, start, end in zip(
                    self.param_groups[0]["params"], self._offsets, self._offsets[1:]
                ):
                    p.data.copy_(vec[start:end].view_as(p))

    def step(
        self,
        *,
        closure_main: Callable[[bool], Tensor],
        closure_d: Callable[[bool], Tensor],
        hNk,
        **_,
    ) -> float:
        # Reset for external
        self.inc_batch_size = False
        self.move_to_next_batch = True

        st = self.state
        # record current flat parameters
        wk = self._flat_params_fn()

        # evaluate objective and gradient
        fN_old = _["loss"] if "loss" in _ else closure_main(compute_grad=True)
        g = _["grad"] if "grad" in _ else self._flat_grads_fn()
        fD_old = closure_d(compute_grad=True)
        g_bar = self._flat_grads_fn()

        # update SR1 memory
        if st["prev_s"] is not None:
            self.hess.update_memory(st["prev_s"], g - st["prev_g"])

        gn = torch.norm(g, p=self.norm_type)
        if self.second_order and len(self.hess._S) > 0:
            print("(INFO) Using second-order ASNTR step.")
            # pred = -(g*p + 0.5*p*B*p)
            step, pred_red = solve_tr_second_order(
                g, gn, self.delta, self.hess, self.obs, self.tol
            )
        else:
            print("(INFO) Using first-order ASNTR step.")
            # pred = -g*p
            step, pred_red = solve_tr_first_order(g, gn, self.delta, self.tol)

        # Since pred is the classical predicted TR reduction, here we multiply it by -1 
        # to abide by Q_k(p) specified in Equation (10) in the paper
        pred *= -1

        # trial update
        self._unflatten_update(wk + step)
        with torch.no_grad():
            fN_new = closure_main(compute_grad=False)
            fD_new = closure_d(compute_grad=False)

        # ratios
        tk = self.c_1 / ((self.k + 1) ** self.alpha)
        ttilde_k = self.c_2 / ((self.k + 1) ** self.alpha)

        print(
            f"abs(hNk): {hNk:.4f}, tol: {self.tol:.4f}, t{self.k} = {tk:.4f}, ttilde_{self.k} = {ttilde_k:.4f}"
        )

        if abs(float(pred_red)) < self.tol:
            rho_N = float("inf")
        else:
            rho_N = (fN_old - fN_new + tk * self.delta) / pred_red

        pred_red_d = -g_bar.dot(step)
        if abs(float(pred_red_d)) < self.tol:
            rho_D = float("inf")
        else:
            rho_D = (fD_old - fD_new + ttilde_k * self.delta) / pred_red_d

        print(f"pred_red = {pred_red:.4f}, pred_red_d = {pred_red_d:.4f}")
        print(f"rho_N = {rho_N:.4f}, rho_D = {rho_D:.4f}")

        if abs(hNk) > self.tol:
            accepted = rho_N >= self.eta and rho_D >= self.nu

            if gn < self.tol * hNk:
                self.inc_batch_size = True
                self.move_to_next_batch = True
            else:
                if rho_D < self.nu:
                    self.inc_batch_size = True
                    self.move_to_next_batch = True
                else:
                    if rho_N < self.eta:
                        self.inc_batch_size = False
                        self.move_to_next_batch = False
                    else:
                        self.inc_batch_size = False
                        self.move_to_next_batch = True
        else:
            accepted = rho_N >= self.eta

        print(f"Increase batch size for next step?: {self.inc_batch_size}")
        print(f"Move to next batch for next step?: {self.move_to_next_batch}")

        if accepted:
            print("(ASNTR) Step accepted.")
            st["prev_s"] = step.clone().detach()
            st["prev_g"] = g.clone().detach()
        else:
            print("(ASNTR) Step rejected, reverting to previous parameters.")
            self._unflatten_update(wk)
            st["prev_s"] = None
            st["prev_g"] = None

        # adjust delta
        if rho_N < self.eta_1:
            self.delta *= self.tau_1
        elif (
            rho_N > self.eta_2
            and torch.norm(step, p=self.norm_type) > self.tau_2 * self.delta
        ):
            self.delta = min(self.delta * self.tau_3, self.max_delta)
        self.delta = max(self.min_delta, min(self.delta, self.max_delta))

        self.k += 1
        return fN_new if accepted else fN_old
