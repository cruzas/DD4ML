from __future__ import annotations
import math
from typing import Callable, Iterable, Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

from dd4ml.solvers.obs import OBS
from .lsr1 import LSR1
from dd4ml.utility.optimizer_utils import solve_tr_first_order, solve_tr_second_order


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
            raise ValueError("Optimiser got an empty parameter list")

        # Store hyperparameters in defaults
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

        # Determine device/dtype from first parameter
        first_param = (
            param_list[0]["params"][0]
            if isinstance(param_list[0], dict)
            else param_list[0]
        )
        device = first_param.device
        dtype = first_param.dtype

        # Compute shapes and offsets for flattening
        shapes: list[torch.Size] = []
        offsets = [0]
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            shapes.append(p.shape)
            offsets.append(offsets[-1] + n)
        total_size = offsets[-1]
        self._shapes = shapes
        self._offsets = offsets

        # Preallocate flat buffers for parameters, gradient, previous step
        st = self.state
        st["flat_wk"] = torch.zeros(total_size, device=device, dtype=dtype)
        st["flat_gk"] = torch.zeros_like(st["flat_wk"])
        st["flat_vk"] = torch.zeros_like(st["flat_wk"])
        st["prev_grad"] = None

        # Cache a tolerance tensor to avoid recreating it repeatedly
        st["tol_tensor"] = torch.tensor(tol, device=device, dtype=dtype)

        # OBS solver for trust-region subproblem
        self.obs = OBS()
        self.hess = LSR1(
            gamma=gamma,
            memory_length=mem_length,
            device=device,
            dtype=dtype,
        )
        
        # Distributed synchronisation flags
        self.sync = bool(sync)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def _avg_scalar(self, value: Tensor) -> Tensor:
        """
        Rank-0 reduce-average and broadcast for bit-exact synchronisation.
        If `sync` is False or `world_size == 1`, return `value` unchanged.
        """
        if not (self.sync and self.world_size > 1):
            return value
        # Cast to float64 once and reuse
        buf = value.detach().to(torch.float64)
        dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            buf /= self.world_size
        dist.broadcast(buf, src=0)
        return buf.to(value.dtype)

    def _flatten_params(self) -> Tensor:
        """
        Copy each parameter tensor into the flat buffer `flat_wk`.
        """
        buf = self.state["flat_wk"]
        for p, start, end in zip(
            self.param_groups[0]["params"],
            self._offsets,
            self._offsets[1:],
        ):
            buf[start:end].copy_(p.data.view(-1))
        return buf

    def _flatten_grads(self) -> Tensor:
        """
        Copy each gradient tensor into the flat buffer `flat_gk`.
        """
        buf = self.state["flat_gk"]
        for p, start, end in zip(
            self.param_groups[0]["params"],
            self._offsets,
            self._offsets[1:],
        ):
            buf[start:end].copy_(p.grad.view(-1))
        return buf

    def _unflatten_update(self, vec: Tensor) -> None:
        """
        Scatter the flat update vector `vec` back into each parameter tensor.
        """
        with torch.no_grad():
            for p, start, end in zip(
                self.param_groups[0]["params"],
                self._offsets,
                self._offsets[1:],
            ):
                p.data.copy_(vec[start:end].view_as(p))

    def _evaluate_function_and_gradient(
        self,
        wk: Tensor,
        p: Tensor,
        alpha: float,
        closure: Callable,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Move to trial point: w = wk + alpha * p,
        evaluate loss and gradient, return (loss, flat_grad, directional_derivative).
        Synchronise scalars if in distributed mode.
        """
        # Update parameters to trial point
        self._unflatten_update(wk + alpha * p)

        # Compute loss and gradient via closure
        loss = closure(compute_grad=True)
        grad = self._flatten_grads()
        
        # Synchronise if needed
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
        """
        Perform zoom phase of strong Wolfe line search, narrowing interval [alpha_lo, alpha_hi].
        """
        tol_tensor = self.state["tol_tensor"]
        for _ in range(max_iter):
            # Interpolate trial alpha
            if _ == 0:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
            else:
                denom = 2 * (phi_hi - phi_lo - dphi_lo * (alpha_hi - alpha_lo))
                cond = torch.abs(denom) > self.defaults["tol"]
                if not cond:
                    interp = 0.5 * (alpha_lo + alpha_hi)
                else:
                    interp = alpha_lo - dphi_lo * (alpha_hi - alpha_lo).square() / denom
                safe_low = alpha_lo + 0.1 * (alpha_hi - alpha_lo)
                safe_high = alpha_hi - 0.1 * (alpha_hi - alpha_lo)
                alpha_j = torch.where(
                    cond,
                    torch.clamp(interp, safe_low, safe_high),
                    0.5 * (alpha_lo + alpha_hi),
                )

            # Evaluate at candidate α_j
            phi_j, grad_j, dphi_j = self._evaluate_function_and_gradient(
                wk, p, alpha_j, closure
            )

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

            # Gap tolerance check
            if torch.abs(alpha_hi - alpha_lo) < tol_tensor:
                break

        # Return best lower bound if max_iter exceeded
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
        Find step size satisfying strong Wolfe conditions. Returns (alpha, phi, grad).
        """
        tol_tensor = self.state["tol_tensor"]
        alpha_prev = torch.tensor(0.0, device=wk.device)
        phi_prev = phi_0
        dphi_prev = dphi_0
        alpha_i = torch.tensor(alpha_0, device=wk.device)

        for i in range(max_iter):
            phi_i, grad_i, dphi_i = self._evaluate_function_and_gradient(
                wk, p, alpha_i.item(), closure
            )

            cond1 = phi_i > phi_0 + c1 * alpha_i * dphi_0
            cond2 = (i > 0) and (phi_i >= phi_prev)
            if cond1 or cond2:
                return self._zoom(
                    wk,
                    p,
                    alpha_prev,
                    alpha_i,
                    phi_prev,
                    phi_i,
                    dphi_prev,
                    phi_0,
                    dphi_0,
                    closure,
                    c1,
                    c2,
                )

            if torch.abs(dphi_i) <= -c2 * dphi_0:
                return alpha_i, phi_i, grad_i

            if dphi_i >= 0:
                return self._zoom(
                    wk,
                    p,
                    alpha_i,
                    alpha_prev,
                    phi_i,
                    phi_prev,
                    dphi_i,
                    phi_0,
                    dphi_0,
                    closure,
                    c1,
                    c2,
                )

            alpha_prev, phi_prev, dphi_prev = alpha_i, phi_i, dphi_i
            # Double step but cap at alpha_max
            alpha_i = torch.min(2 * alpha_i, torch.tensor(alpha_max, device=wk.device))

            if alpha_i >= alpha_max:
                break

        # Fallback to previous iterate if no step found
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
        max_iter: int = 10,
    ) -> Tuple[float, float, Tensor]:
        """
        Simple backtracking line search as fallback. Returns (alpha, φ(alpha), grad).
        """
        alpha = alpha_0
        tol_tensor = self.state["tol_tensor"]
        for _ in range(max_iter):
            phi_alpha, grad_alpha, dphi_alpha = self._evaluate_function_and_gradient(
                wk, p, alpha, closure
            )
            armijo_ok = phi_alpha <= phi_0 + c1 * alpha * dphi_0
            curvature_ok = torch.abs(dphi_alpha) <= c2 * torch.abs(dphi_0)
            if armijo_ok and curvature_ok:
                return alpha, phi_alpha, grad_alpha
            alpha *= 0.5
            if alpha < tol_tensor.item():
                break
        return 0.0, phi_0, self._flatten_grads()

    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[float, float]:
        """
        Performs a single optimisation step.
        Returns (new_loss, flat_gradient) on all ranks (scalar loss and flat grad vector).
        """
        # Get precomputed loss/grad or compute afresh
        loss = _["precomp_loss"] if "precomp_loss" in _ else closure(compute_grad=True)
        g = _["precomp_grad"] if "precomp_grad" in _ else self._flatten_grads()
        gn = torch.norm(g, p=self.defaults["norm_type"])
        if self.sync and self.world_size > 1:
            loss = self._avg_scalar(loss)
            dist.broadcast(g, src=0)
            gn = self._avg_scalar(gn)
        if gn <= self.defaults["tol"]:
            return loss.item(), g  # No sufficient gradient, skip update

        # Flatten current parameters once and reuse
        wk_flat = self._flatten_params()
        wk = wk_flat.clone()  # Preserve copy for TR and memory updates
        st = self.state
        sec = self.defaults["second_order"]

        # Update LSR1 memory if second order is enabled
        if sec and st["prev_grad"] is not None:
            sk = wk - st["old_wk"]
            yk = g - st["prev_grad"]
            if sk.norm() > self.defaults["tol"] and yk.norm() > self.defaults["tol"]:
                self.hess.update_memory(sk, yk)
        st["old_wk"], st["prev_grad"] = wk.clone(), g.clone()

        # Solve trust-region subproblem
        if sec and len(self.hess._S) > 0:
            p_star, pred = solve_tr_second_order(g, gn, self.defaults["delta"], self.hess, self.obs, self.defaults["tol"])
        else:
            p_star, pred = solve_tr_first_order(g, gn, self.defaults["delta"], self.defaults["tol"])

        # Momentum-like update for vk; reuse existing flat buffer
        vk = st["flat_vk"]
        vk.mul_(self.defaults["mu"]).add_(wk - st["old_wk"])
        # Bound vk to trust-region radius
        vk_norm_sq = vk.dot(vk)
        if vk_norm_sq > 0.0:
            vk_norm = vk_norm_sq.sqrt()
            scale = min(1.0, self.defaults["delta"] / vk_norm)
            vk.mul_(scale)
        # Combine step and bound to TR radius
        p_comb = p_star + vk
        p_comb_norm_sq = p_comb.dot(p_comb)
        if p_comb_norm_sq > 0.0:
            p_comb_norm = p_comb_norm_sq.sqrt()
            scale = min(1.0, self.defaults["delta"] / p_comb_norm)
            p_comb.mul_(scale)
        st["flat_vk"] = vk.clone()  # Store updated vk for next iteration

        # Prepare for line search
        phi_0 = loss
        dphi_0 = g.dot(p_comb)

        # Strong Wolfe line search (preferred)
        alpha, new_loss, new_g = self._strong_wolfe_line_search(
            wk,
            p_comb,
            phi_0,
            dphi_0,
            closure,
            alpha_0=self.defaults["lr"],
            c1=self.defaults["c1"],
            c2=self.defaults["c2"],
            alpha_max=self.defaults["alpha_max"],
            max_iter=self.defaults["max_wolfe_iter"],
        )

        # Apply final step
        p_step = alpha * p_comb
        self._unflatten_update(wk + p_step)

        # Update trust-region radius based on ρ = (f(new)−f(old)) / predicted
        rho = (
            (new_loss - loss) / pred if (alpha > 0 and pred < 0) else 0.0
        )
        s_norm_sq = p_step.dot(p_step)
        s_norm = math.sqrt(s_norm_sq.item())
        delta_old = self.defaults["delta"]
        if rho < self.defaults["tau_2"]:
            # Shrink trust region
            delta_new = min(
                self.defaults["nu_1"] * delta_old,
                self.defaults["nu_2"] * s_norm**2,
            )
        elif rho >= self.defaults["tau_3"] and s_norm >= self.defaults["nu_3"] * delta_old:
            # Expand trust region
            delta_new = self.defaults["nu_4"] * delta_old
        else:
            delta_new = delta_old
        # Clip to [min_delta, max_delta]
        self.defaults["delta"] = max(
            self.defaults["min_delta"],
            min(delta_new, self.defaults["max_delta"]),
        )

        return new_loss, new_g
