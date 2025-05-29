from __future__ import annotations
import math
from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from dd4ml.solvers.obs import OBS
from .lsr1 import LSR1


class LSSR1_TR(Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 1.0,
        delta: float = 1.0,
        min_delta: float = 1e-3,
        max_delta: float = 2.0,
        gamma: float = 1e-3,
        mem_length: int = 10,
        mu: float = 0.9,
        tau_1: float = 0.1,
        tau_2: float = 0.25,
        tau_3: float = 0.75,
        nu_1: float = 0.25,
        nu_2: float = 0.5,
        nu_3: float = 0.8,
        nu_4: float = 1.2,
        tol: float = 1e-8,
        norm_type: int = 2,
    ):
        param_list = list(params)
        if not param_list:
            raise ValueError("Optimizer got an empty parameter list")

        defaults = dict(
            lr=lr, delta=delta, min_delta=min_delta, max_delta=max_delta,
            gamma=gamma, mem_length=mem_length, mu=mu,
            tau_1=tau_1, tau_2=tau_2, tau_3=tau_3,
            nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4,
            tol=tol, norm_type=norm_type,
        )
        super().__init__(param_list, defaults)

        # preserve your p0 logic
        p0 = (
            param_list[0]["params"][0]
            if isinstance(param_list[0], dict)
            else param_list[0]
        )

        # record shapes & offsets
        shapes, offsets = [], [0]
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            shapes.append(p.shape)
            offsets.append(offsets[-1] + n)
        total = offsets[-1]
        self._shapes = shapes
        self._offsets = offsets

        # persistent flat buffers
        st = self.state
        st["flat_wk"] = torch.zeros(total, device=p0.device, dtype=p0.dtype)
        st["flat_gk"] = torch.zeros_like(st["flat_wk"])
        st["flat_vk"] = torch.zeros_like(st["flat_wk"])
        st["prev_grad"] = None

        # Hessian machinery
        self.obs = OBS()
        self.hess = LSR1(
            gamma=gamma, memory_length=mem_length,
            device=p0.device, dtype=p0.dtype,
        )

    def _flatten_params(self) -> Tensor:
        buf = self.state["flat_wk"]
        for p, start, end in zip(self.param_groups[0]["params"], self._offsets, self._offsets[1:]):
            buf[start:end].copy_(p.data.view(-1))
        return buf

    def _flatten_grads(self) -> Tensor:
        buf = self.state["flat_gk"]
        for p, start, end in zip(self.param_groups[0]["params"], self._offsets, self._offsets[1:]):
            buf[start:end].copy_(p.grad.view(-1))
        return buf

    def _unflatten_update(self, vec: Tensor) -> None:
        with torch.no_grad():
            for p, start, end in zip(self.param_groups[0]["params"], self._offsets, self._offsets[1:]):
                p.data.copy_(vec[start:end].view_as(p))

    def _solve_tr_first_order(self, g: Tensor, gn: float, delta: float) -> Tuple[Tensor, float]:
        if gn <= self.defaults["tol"]:
            return torch.zeros_like(g), 0.0
        step = -g * (delta / gn)
        predicted = delta * gn
        return step, float(predicted)

    def _solve_tr_second_order(self, g: Tensor, gn: float, delta: float) -> Tuple[Tensor, float]:
        if gn <= self.defaults["tol"]:
            return torch.zeros_like(g), 0.0
        self.hess.precompute()
        step = -self.obs.solve_tr_subproblem(
            g,
            torch.tensor(delta, device=g.device, dtype=g.dtype),
            self.hess.gamma,
            self.hess.Psi,
            self.hess.Minv,
        )
        g_dot_p = torch.dot(g, step)
        p_B_p = torch.dot(step, self.hess.B(step))
        predicted = -(g_dot_p + 0.5 * p_B_p)
        return step, float(predicted)

    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[float, float]:
        loss = float(closure())
        g = self._flatten_grads()
        gn = torch.norm(g, p=self.defaults["norm_type"]).item()
        if gn <= self.defaults["tol"]:
            return loss, gn

        wk = self._flatten_params().clone()
        st = self.state

        # update L-SR1 memory
        if st["prev_grad"] is not None:
            sk = wk - st["old_wk"]
            yk = g - st["prev_grad"]
            if sk.norm() > self.defaults["tol"] and yk.norm() > self.defaults["tol"]:
                self.hess.update_memory(sk, yk)
        st["old_wk"], st["prev_grad"] = wk.clone(), g.clone()

        # choose TR solver
        if len(self.hess._S) > 0:
            p_star, pred = self._solve_tr_second_order(g, gn, self.defaults["delta"])
        else:
            p_star, pred = self._solve_tr_first_order(g, gn, self.defaults["delta"])

        # momentum grafting
        vk = st["flat_vk"]
        vk.mul_(self.defaults["mu"]).add_(wk - st["old_wk"])
        if vk.norm() > 0:
            vk.mul_(min(1.0, self.defaults["delta"] / vk.norm()))
        p_comb = p_star + vk
        if p_comb.norm() > 0:
            p_comb.mul_(min(1.0, self.defaults["delta"] / p_comb.norm()))
        st["flat_vk"] = vk.clone()

        # Armijo line search
        alpha = self.defaults["lr"]
        c1 = 1e-4
        grad_dir = g.dot(p_comb).item()
        for _ in range(20):
            self._unflatten_update(wk + alpha * p_comb)
            new_loss = float(closure())
            if new_loss <= loss + c1 * alpha * grad_dir:
                break
            alpha = max(0.5 * alpha, self.defaults["min_delta"])
        else:
            alpha = 0.0
            self._unflatten_update(wk)

        # final step and trust-region update
        p_step = alpha * p_comb
        self._unflatten_update(wk + p_step)

        rho = (new_loss - loss) / pred if (alpha > 0 and pred < 0) else 0.0
        s_norm = p_step.norm().item()
        delta_old = self.defaults["delta"]
        if rho < self.defaults["tau_2"]:
            delta_new = min(self.defaults["nu_1"] * delta_old, self.defaults["nu_2"] * s_norm**2)
        elif rho >= self.defaults["tau_3"] and s_norm >= self.defaults["nu_3"] * delta_old:
            delta_new = self.defaults["nu_4"] * delta_old
        else:
            delta_new = delta_old
        self.defaults["delta"] = max(self.defaults["min_delta"], min(delta_new, self.defaults["max_delta"]))

        return loss, gn
