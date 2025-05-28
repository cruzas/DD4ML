from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from dd4ml.solvers.obs import OBS

from .hessian_approx import LSR1


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _concat_params(params: Iterable[Tensor]) -> Tensor:
    return torch.cat([p.detach().flatten() for p in params])


def _concat_grads(params: Iterable[Tensor]) -> Tensor:
    return torch.cat([p.grad.flatten() for p in params])


def _set_param_vector(params: Iterable[Tensor], vec: Tensor) -> None:
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(vec[offset : offset + numel].view_as(p))
        offset += numel


# --------------------------------------------------------------------------- #
# L-SSR1-TR Optimiser                                                         #
# --------------------------------------------------------------------------- #
class LSSR1_TR(Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 0.1,  # current line-search step length (α)
        delta: float = 1.0,  # current trust-region radius (Δk in Alg. 3)
        gamma: float = 1e-3,
        mem_length: int = 10,
        mu: float = 0.9,
        tau_1: float = 0.1,
        tau_2: float = 0.25,
        tau_3: float = 0.75,
        nu_1: float = 0.25,
        nu_2: float = 0.5,
        nu_3: float = 0.8,
        nu_4: float = 2.0,
        tol: float = 1e-8,
        norm_type: int = 2,  # norm type for gradient and step length
    ):
        # (range checks unchanged …)

        param_list = list(params)
        if not param_list:
            raise ValueError("Optimizer got an empty parameter list")

        # ---- canonical defaults expected by PyTorch -------------------
        defaults = dict(
            lr=lr,
            delta=delta,
            gamma=gamma,
            mem_length=mem_length,
            mu=mu,
            tol=tol,
            tau_1=tau_1,
            tau_2=tau_2,
            tau_3=tau_3,
            nu_1=nu_1,
            nu_2=nu_2,
            nu_3=nu_3,
            nu_4=nu_4,
            norm_type=norm_type,
        )
        super().__init__(param_list, defaults)

        # representative tensor for device/dtype
        p0 = (
            param_list[0]["params"][0]
            if isinstance(param_list[0], dict)
            else param_list[0]
        )

        self.obs = OBS()
        self.hess = LSR1(
            gamma=gamma,
            memory_length=mem_length,
            device=p0.device,
            dtype=p0.dtype,
        )

        self.state["wk"] = None
        self.state["prev_grad"] = None
        self.state["vk"] = torch.zeros(1)

    # ---------------- sub-problem solvers ----------------------------- #
    def _solve_tr_first_order(
        self, g: torch.Tensor, delta: float
    ) -> Tuple[torch.Tensor, float]:
        g_norm = torch.norm(g, p=self.defaults["norm_type"])
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

    # ------------------------------------------------------------------ #
    # Main optimisation step                                             #
    # ------------------------------------------------------------------ #
    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[float, float]:
        params = self.param_groups[0]["params"]

        loss = float(_["precomp_loss"]) if "precomp_loss" in _ else closure()
        g = _["precomp_grad"] if "precomp_grad" in _ else _concat_grads(params)

        g_norm = float(g.norm())
        if g_norm <= self.defaults["tol"]:
            return float(loss), g_norm

        wk = _concat_params(params)
        st = self.state
        if st["wk"] is None:
            st["wk"] = wk.clone()
        if st["vk"].numel() != wk.numel():
            st["vk"] = torch.zeros_like(wk)

        # -- update L-SR1 pairs -----------------------------------------
        if st["prev_grad"] is not None:
            self.hess.update_memory(wk - st["wk"], g - st["prev_grad"])
        st["wk"], st["prev_grad"] = wk.clone(), g.clone()

        # -- trust-region sub-problem -----------------------------------
        solve = (
            self._solve_tr_second_order
            if len(self.hess._S) > 0  # type: ignore[attr-defined]
            else self._solve_tr_first_order
        )

        p_star, pred = solve(g, self.defaults["delta"])

        # -- momentum grafting ------------------------------------------
        vk = st["vk"] * self.defaults["mu"] + (wk - st["wk"])
        if vk.norm() > 0:
            vk *= min(1.0, self.defaults["delta"] / vk.norm())
        p_comb = p_star + vk
        if p_comb.norm() > 0:
            p_comb *= min(1.0, self.defaults["delta"] / p_comb.norm())
        st["vk"] = vk.clone()

        # -- Wolfe line-search ------------------------------------------
        alpha = self.defaults["lr"]
        c1 = 1e-4
        orig_loss = float(loss)
        grad_dot_dir = g.dot(p_comb)

        for _ in range(10):
            _set_param_vector(params, wk + alpha * p_comb)
            new_loss = float(closure())
            if new_loss <= orig_loss + c1 * alpha * grad_dot_dir:  # Armijo
                break
            alpha *= 0.5
        else:  # no break
            alpha = 0.0
            _set_param_vector(params, wk)

        p_alpha = alpha * p_comb  # accepted step

        # -- trust-region radius update ---------------------------------
        delta_old = self.defaults["delta"]
        rho = 0.0
        if alpha > 0 and pred < 0:
            rho = (new_loss - orig_loss) / pred

        s_norm = p_alpha.norm()
        if rho < self.defaults["tau_2"]:
            delta_new = min(
                self.defaults["nu_1"] * delta_old, self.defaults["nu_2"] * (s_norm**2)
            )
        elif (
            rho >= self.defaults["tau_3"]
            and s_norm >= self.defaults["nu_3"] * delta_old
        ):
            delta_new = self.defaults["nu_4"] * delta_old
        else:
            delta_new = delta_old
        self.defaults["delta"] = max(delta_new, self.defaults["tol"])

        # -- prepare return values --------------------------------------
        grad_norm = g_norm if alpha == 0.0 else float(_concat_grads(params).norm())

        # expose the step length through every param-group (for schedulers)
        for group in self.param_groups:
            group["lr"] = alpha
        self.defaults["lr"] = alpha

        return float(loss), g
