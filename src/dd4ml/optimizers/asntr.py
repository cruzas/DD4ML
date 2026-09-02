from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce

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
        dogleg: bool = False,
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
        self.dogleg = bool(dogleg)  # dogleg is only used in second-order mode
        if self.dogleg and not self.second_order:
            raise ValueError("Dogleg is only applicable in second-order mode")

        # SR1 memory and OBS solver
        self.hess = LSR1(
            gamma=gamma, memory_length=mem_length, device=self.device, tol=self.tol
        )
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
        params = self.param_groups[0]["params"]
        shapes: list[torch.Size] = []
        offsets = [0]
        for p in params:
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
            # Take the dtype from the parameters rather than the global default.
            # Every step stages parameters and gradients through these buffers,
            # so allocating float32 here silently truncated a float64 model on
            # each flatten/unflatten round trip. Mixed-precision parameters are
            # promoted to the widest dtype present for the same reason.
            buf_dtype = reduce(torch.promote_types, (p.dtype for p in params))
            buf = torch.zeros(total_size, device=self.device, dtype=buf_dtype)
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
        return buf.clone()

    def _flatten_grads(self) -> Tensor:
        buf = self.state["flat_gk"]
        for p, start, end in zip(
            self.param_groups[0]["params"], self._offsets, self._offsets[1:]
        ):
            buf[start:end].copy_(p.grad.view(-1))
        return buf.clone()

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
            # pred_red = -(g*p + 0.5*p*B*p)
            step, pred_red = solve_tr_second_order(
                gradient=g,
                grad_norm=gn,
                trust_radius=self.delta,
                lsr1_hessian=self.hess,
                obs_solver=self.obs,
                tol=self.tol,
                dogleg=self.dogleg,
            )
        else:
            print("(INFO) Using first-order ASNTR step.")
            # pred_red = -g*p
            step, pred_red = solve_tr_first_order(g, gn, self.delta, self.tol)

        # solve_tr_* returns the classical (positive) predicted reduction. Negate it
        # to obtain Q_k(p_k) as defined in Eq. (4) of the paper, which is negative.
        pred_red *= -1

        # trial update
        self._unflatten_update(wk + step)
        with torch.no_grad():
            fN_new = closure_main(compute_grad=False)
            fD_new = closure_d(compute_grad=False)

        # ratios
        tk = self.c_1 / ((self.k + 1) ** self.alpha)
        ttilde_k = self.c_2 / ((self.k + 1) ** self.alpha)

        # print(
        #     f"abs(hNk): {hNk:.4f}, tol: {self.tol:.4f}, t{self.k} = {tk:.4f}, ttilde_{self.k} = {ttilde_k:.4f}"
        # )

        # Non-monotone reference value for the N-sample, Eq. (7):
        #   r_{N_k} = f_{N_k}(w_k) + t_k * delta_k
        # and the agreement ratio, Eq. (6):
        #   rho_{N_k} = (f_{N_k}(w_t) - r_{N_k}) / Q_k(p_k)
        # Q_k(p_k) < 0 by the Cauchy-decrease condition Eq. (5), so a trial point
        # that improves on the reference value gives rho_N > 0.
        if abs(float(pred_red)) < self.tol:
            rho_N = float("inf")
        else:
            r_Nk = fN_old + tk * self.delta
            rho_N = (fN_new - r_Nk) / pred_red

        # Additional-sampling agreement ratio, Eq. (9):
        #   rho_{D_k} = (f_{D_k}(w_t) - r_{D_k}) / L_k(-g_bar_k)
        # with the linear model L_k(v) = v^T g_bar_k, so the denominator is
        # L_k(-g_bar_k) = -||g_bar_k||^2 <= 0, and Eq. (10):
        #   r_{D_k} = f_{D_k}(w_k) + delta_k * ttilde_k
        lin_red_d = -g_bar.dot(g_bar)
        if abs(float(lin_red_d)) < self.tol:
            rho_D = float("inf")
        else:
            r_Dk = fD_old + self.delta * ttilde_k
            rho_D = (fD_new - r_Dk) / lin_red_d

        # print(f"rho_N = {rho_N:.4f}, rho_D = {rho_D:.4f}")

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

        # print(f"Increase batch size for next step?: {self.inc_batch_size}")
        # print(f"Move to next batch for next step?: {self.move_to_next_batch}")

        if accepted:
            print("(ASNTR) Step accepted.")
            st["prev_s"] = step.clone()
            st["prev_g"] = g.clone()
        else:
            print("(ASNTR) Step rejected, reverting to previous parameters.")
            self._unflatten_update(wk)
            st["prev_s"] = None
            st["prev_g"] = None

        # adjust delta
        if rho_N < self.eta_1:
            self.delta = max(self.min_delta, self.delta * self.tau_1)
        elif (
            rho_N > self.eta_2
            and torch.norm(step, p=self.norm_type) > self.tau_2 * self.delta
        ):
            self.delta = min(self.max_delta, self.delta * self.tau_3)

        self.k += 1
        return fN_new if accepted else fN_old
