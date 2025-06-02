from __future__ import annotations
import math
from typing import Callable, Iterable, Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

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
        second_order: bool = True,
        mem_length: int = 10,
        max_wolfe_iter: int = 10,
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
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_max: float = 10.0,
        sync: bool = False,
    ):
        param_list = list(params)
        if not param_list:
            raise ValueError("Optimizer got an empty parameter list")

        defaults = dict(
            lr=lr,
            delta=delta,
            min_delta=min_delta,
            max_delta=max_delta,
            gamma=gamma,
            second_order=second_order,
            mem_length=mem_length,
            mu=mu,
            tau_1=tau_1,
            tau_2=tau_2,
            tau_3=tau_3,
            nu_1=nu_1,
            nu_2=nu_2,
            nu_3=nu_3,
            nu_4=nu_4,
            tol=tol,
            norm_type=norm_type,
            max_wolfe_iter=max_wolfe_iter,
            c1=c1,
            c2=c2,
            alpha_max=alpha_max,
        )
        super().__init__(param_list, defaults)

        p0 = (
            param_list[0]["params"][0]
            if isinstance(param_list[0], dict)
            else param_list[0]
        )

        shapes, offsets = [], [0]
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            shapes.append(p.shape)
            offsets.append(offsets[-1] + n)
        total = offsets[-1]
        self._shapes = shapes
        self._offsets = offsets

        st = self.state
        st["flat_wk"] = torch.zeros(total, device=p0.device, dtype=p0.dtype)
        st["flat_gk"] = torch.zeros_like(st["flat_wk"])
        st["flat_vk"] = torch.zeros_like(st["flat_wk"])
        st["prev_grad"] = None

        self.obs = OBS()
        self.hess = LSR1(
            gamma=gamma,
            memory_length=mem_length,
            device=p0.device,
            dtype=p0.dtype,
        )
        
        # distributed sync control
        self.sync = bool(sync)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def _avg_scalar(self, value: Tensor) -> Tensor:
        """
        Rank-0 reduce-average and broadcast for identical bit patterns.
        Input: 0-dim tensor
        Output: 0-dim tensor (float64→float32 if needed)
        If `self.sync` is `False`, the input tensor is returned unchanged.
        """
        if not (self.sync and self.world_size > 1):
            return value
        # cast to float64 for reduction
        buf = value.detach().to(torch.float64)
        dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            buf /= self.world_size
        dist.broadcast(buf, src=0)
        return buf.to(value.dtype)

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
        return step, predicted

    def _solve_tr_second_order(
        self, g: Tensor, gn: float, delta: float
    ) -> Tuple[Tensor, float]:
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
        return step, predicted.item()

    def _evaluate_function_and_gradient(
        self,
        wk: Tensor,
        p:  Tensor,
        alpha: float,
        closure: Callable,
    ) -> Tuple[float, Tensor, float]:
        """
        - Move to trial point: w = wk + alpha*p
        - Compute loss = closure() → Tensor
        - Compute flat gradient = self._flatten_grads()
        - Return (loss_avg, grad_broadcast, deriv_avg) as 0-dim Tensors
        """
        # Move to trial point
        self._unflatten_update(wk + alpha * p)

        # Objective and gradient
        loss = closure(compute_grad=True)
        grad = self._flatten_grads()
        
        # synchronize scalars & gradient for bit-exact behaviour
        if self.sync and self.world_size > 1:
            loss = self._avg_scalar(loss)
            deriv = grad.dot(p)
            deriv = self._avg_scalar(deriv)
            dist.broadcast(grad, src=0)
        else:
            deriv = grad.dot(p)
        return loss, grad, deriv

    def _zoom(
        self,
        wk: Tensor,
        p: Tensor,
        alpha_lo: Tensor,
        alpha_hi: Tensor,
        phi_lo: Tensor,
        phi_hi: Tensor,
        dphi_lo: Tensor,
        phi_0: Tensor,
        dphi_0: Tensor,
        closure: Callable,
        c1: float,
        c2: float,
        max_iter: int = 20,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        tol = torch.tensor(self.defaults["tol"], device=wk.device)
        for i in range(max_iter):
            # candidate α_j
            if i == 0:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
            else:
                denom = 2 * (phi_hi - phi_lo - dphi_lo * (alpha_hi - alpha_lo))
                if denom == 0:
                    interp = 0.5 * (alpha_lo + alpha_hi)  # Fallback to midpoint
                else:
                    cond = torch.abs(denom) > 1e-12
                    interp = alpha_lo - dphi_lo * (alpha_hi - alpha_lo) ** 2 / denom
                safe_low = alpha_lo + 0.1 * (alpha_hi - alpha_lo)
                safe_high = alpha_hi - 0.1 * (alpha_hi - alpha_lo)
                alpha_j = torch.where(
                    cond,
                    torch.clamp(interp, safe_low, safe_high),
                    0.5 * (alpha_lo + alpha_hi),
                )

            # trial evaluation
            phi_j, grad_j, dphi_j = self._evaluate_function_and_gradient(
                wk, p, alpha_j, closure
            )

            # update interval
            armijo = phi_j > phi_0 + c1 * alpha_j * dphi_0
            if armijo or (phi_j >= phi_lo):
                alpha_hi, phi_hi = alpha_j, phi_j
            else:
                strong_wolfe = torch.abs(dphi_j) <= -c2 * dphi_0
                if strong_wolfe:
                    return alpha_j, phi_j, grad_j
                switch = dphi_j * (alpha_hi - alpha_lo) >= 0
                if switch:
                    alpha_hi, phi_hi = alpha_lo, phi_lo
                alpha_lo, phi_lo, dphi_lo = alpha_j, phi_j, dphi_j

            # gap tolerance
            if torch.abs(alpha_hi - alpha_lo) < tol:
                break

        return alpha_lo, phi_lo, self._flatten_grads()

    def _strong_wolfe_line_search(
        self,
        wk: Tensor,
        p: Tensor,
        phi_0: Tensor,
        dphi_0: Tensor,
        closure: Callable,
        alpha_0: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_max: float = 10.0,
        max_iter: int = 10,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns (alpha, phi, grad) as 0-dim Tensors, synchronised across ranks
        if self.sync=True and world_size>1.
        """
        tol = torch.tensor(self.defaults["tol"], device=wk.device)
        alpha_prev = torch.tensor(0.0, device=wk.device)
        phi_prev   = phi_0
        dphi_prev  = dphi_0
        alpha_i    = torch.tensor(alpha_0, device=wk.device)

        for _ in range(max_iter):
            phi_i, grad_i, dphi_i = self._evaluate_function_and_gradient(
                wk, p, alpha_i.item(), closure
            )

            cond1 = phi_i > phi_0 + c1 * alpha_i * dphi_0
            cond2 = (_ > 0) and (phi_i >= phi_prev)
            if cond1 or cond2:
                return self._zoom(
                    wk, p,
                    alpha_prev, alpha_i,
                    phi_prev,  phi_i,
                    dphi_prev, phi_0, dphi_0,
                    closure, c1, c2,
                )

            if torch.abs(dphi_i) <= -c2 * dphi_0:
                return alpha_i, phi_i, grad_i

            if dphi_i >= 0:
                return self._zoom(
                    wk, p,
                    alpha_i, alpha_prev,
                    phi_i,  phi_prev,
                    dphi_i, phi_0, dphi_0,
                    closure, c1, c2,
                )

            alpha_prev, phi_prev, dphi_prev = alpha_i, phi_i, dphi_i
            alpha_i = torch.min(2 * alpha_i, torch.tensor(alpha_max, device=wk.device))

            if alpha_i >= alpha_max:
                break

        # fallback:
        phi_fin, grad_fin, _ = self._evaluate_function_and_gradient(
            wk, p, alpha_prev.item(), closure
        )
        return alpha_prev, phi_fin, grad_fin

    def _backtracking_line_search(
        self,
        wk: Tensor,
        p: Tensor,
        phi_0: float,
        dphi_0: float,
        closure: Callable,
        alpha_0: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 10
    ) -> Tuple[float, float, Tensor]:
        """
        Simple backtracking line search (fallback method).
        
        Returns:
            alpha: step size
            phi_alpha: function value at alpha
            grad_alpha: gradient at alpha
        """
        alpha = alpha_0
        
        min_alpha = 1e-6  # Define a minimum threshold for alpha
        for _ in range(max_iter):
            phi_alpha, grad_alpha, dphi_alpha = self._evaluate_function_and_gradient(wk, p, alpha, closure)
            
            # Check both Wolfe conditions
            armijo_satisfied = phi_alpha <= phi_0 + c1 * alpha * dphi_0
            curvature_satisfied = abs(dphi_alpha) <= c2 * abs(dphi_0)
            
            if armijo_satisfied and curvature_satisfied:
                return alpha, phi_alpha, grad_alpha
            
            alpha *= 0.5
            if alpha < min_alpha:  # Check if alpha is below the minimum threshold
                break
        
        # If no acceptable step found, return zero step
        return 0.0, phi_0, self._flatten_grads()

    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[float, float]:
        loss = _['precomp_loss'] if 'precomp_loss' in _ else closure(compute_grad=True)
        g = _['precomp_grad'] if 'precomp_grad' in _ else self._flatten_grads()
        gn = torch.norm(g, p=self.defaults["norm_type"]).item()
        if gn <= self.defaults["tol"]:
            return loss.item(), gn

        wk = self._flatten_params().clone()
        st = self.state
        sec = self.defaults["second_order"]

        if sec and st["prev_grad"] is not None:
            sk = wk - st["old_wk"]
            yk = g - st["prev_grad"]
            if sk.norm() > self.defaults["tol"] and yk.norm() > self.defaults["tol"]:
                self.hess.update_memory(sk, yk)
        st["old_wk"], st["prev_grad"] = wk.clone(), g.clone()

        if sec and len(self.hess._S) > 0:
            p_star, pred = self._solve_tr_second_order(
                g, gn, self.defaults["delta"]
            )
        else:
            p_star, pred = self._solve_tr_first_order(
                g, gn, self.defaults["delta"]
            )

        vk = st["flat_vk"]
        vk.mul_(self.defaults["mu"]).add_(wk - st["old_wk"])
        if vk.norm() > 0:
            vk.mul_(min(1.0, self.defaults["delta"] / vk.norm()))
        p_comb = p_star + vk
        if p_comb.norm() > 0:
            p_comb.mul_(min(1.0, self.defaults["delta"] / p_comb.norm()))
        st["flat_vk"] = vk.clone()

        # Prepare for line search
        phi_0 = loss
        dphi_0 = g.dot(p_comb)
        
        # Use strong Wolfe line search with zoom
        alpha, new_loss, new_g = self._strong_wolfe_line_search(
            wk, p_comb, phi_0, dphi_0, closure,
            alpha_0=self.defaults["lr"],
            c1=self.defaults["c1"],
            c2=self.defaults["c2"],
            alpha_max=self.defaults["alpha_max"],
            max_iter=self.defaults["max_wolfe_iter"]
        )
    
        p_step = alpha * p_comb
        self._unflatten_update(wk + p_step)

        # Trust region radius update
        rho = (new_loss - loss) / pred if (alpha > 0 and pred < 0) else 0.0
        s_norm = p_step.norm().item()
        delta_old = self.defaults["delta"]
        if rho < self.defaults["tau_2"]:
            delta_new = min(
                self.defaults["nu_1"] * delta_old,
                self.defaults["nu_2"] * s_norm**2,
            )
        elif rho >= self.defaults["tau_3"] and s_norm >= self.defaults["nu_3"] * delta_old:
            delta_new = self.defaults["nu_4"] * delta_old
        else:
            delta_new = delta_old
        self.defaults["delta"] = max(
            self.defaults["min_delta"],
            min(delta_new, self.defaults["max_delta"]),
        )

        return new_loss, new_g