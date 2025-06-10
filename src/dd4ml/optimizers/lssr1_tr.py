from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.solvers.obs import OBS
from dd4ml.utility.optimizer_utils import solve_tr_first_order, solve_tr_second_order

from .lsr1 import LSR1


class LSSR1_TR(Optimizer):
    __name__ = "LSSR1_TR"

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
        max_wolfe_iters: int = 10,
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
        c_1: float = 1e-4,
        c_2: float = 0.9,
        alpha_max: float = 10.0,
        sync: bool = False,
        flat_grads_fn=None,
        flat_params_fn=None,
        flat_params=None,  # only passed by APTS_IP
    ):
        # Ensure at least one parameter is provided
        param_list = list(params)
        if not param_list:
            raise ValueError("Optimiser got an empty parameter list")

        # Only lr remains in defaults
        super().__init__(param_list, {"lr": lr})

        # Assign other hyperparameters as attributes
        self.delta = delta
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.gamma = gamma
        self.second_order = second_order
        self.mem_length = mem_length
        self.max_wolfe_iters = max_wolfe_iters
        self.mu = mu
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu_3 = nu_3
        self.nu_4 = nu_4
        self.tol = tol
        self.norm_type = norm_type
        self.c_1 = c_1
        self.c_2 = c_2
        self.alpha_max = alpha_max

        # Derive device and dtype from the first parameter tensor
        first_param = (
            param_list[0]["params"][0]
            if isinstance(param_list[0], dict)
            else param_list[0]
        )
        device = first_param.device
        dtype = first_param.dtype

        # Compute shapes and offsets to flatten all parameters into a single vector
        shapes: list[torch.Size] = []
        offsets = [0]
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            shapes.append(p.shape)
            offsets.append(offsets[-1] + n)
        total_size = offsets[-1]
        self._shapes = shapes
        self._offsets = offsets

        # Preallocate flat buffers in optimizer state for parameters, gradients, and momentum-like term
        st = self.state
        if flat_params is not None:
            # If flat_params is provided, use it directly
            st["flat_wk"] = WeightParallelizedTensor(
                [torch.zeros_like(t) for t in flat_params.tensor],
                flat_params.backend,
                flat_params.master_group,
                flat_params.rank,
            )
            st["flat_gk"] = WeightParallelizedTensor(
                [torch.zeros_like(t) for t in flat_params.tensor],
                flat_params.backend,
                flat_params.master_group,
                flat_params.rank,
            )
            st["flat_vk"] = WeightParallelizedTensor(
                [torch.zeros_like(t) for t in flat_params.tensor],
                flat_params.backend,
                flat_params.master_group,
                flat_params.rank,
            )
        else:
            st["flat_wk"] = torch.zeros(total_size, device=device, dtype=dtype)
            st["flat_gk"] = torch.zeros_like(st["flat_wk"])
            st["flat_vk"] = torch.zeros_like(st["flat_wk"])
        st["prev_grad"] = None

        # Cache tolerance as tensor to avoid recreating each time
        st["tol_tensor"] = torch.tensor(self.tol, device=device, dtype=dtype)

        # Instantiate OBS solver for trust-region subproblem
        self.obs = OBS()
        # Instantiate limited-memory SR1 Hessian approximation
        self.hess = LSR1(
            gamma=self.gamma,
            memory_length=self.mem_length,
            device=device,
            dtype=dtype,
        )

        # Flags for distributed synchronisation
        self.sync = bool(sync)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self._flat_grads_fn = (
            flat_grads_fn if flat_grads_fn is not None else self._flatten_grads
        )
        self._flat_params_fn = (
            flat_params_fn if flat_params_fn is not None else self._flatten_params
        )

    def _avg_scalar(self, value: Tensor) -> Tensor:
        """
        Perform reduce-average on scalar tensor across ranks, then broadcast.
        If `sync` is False or single process, return value unchanged.
        """
        if not (self.sync and self.world_size > 1):
            return value
        # Use float64 accumulation for precision
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
            if isinstance(vec, WeightParallelizedTensor):
                for p, shard in zip(self.param_groups[0]["params"], vec.tensor):
                    p.data.copy_(shard.view_as(p))
            else:
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
        Move to trial point w = wk + alpha * p, evaluate loss and gradient.
        Return loss value, flattened gradient, and directional derivative along p.
        Synchronise scalars and gradient if in distributed mode.
        """
        # Update parameters to the trial point
        self._unflatten_update(wk + alpha * p)

        # Compute loss and gradients via closure
        loss = closure(compute_grad=True)
        grad = self._flat_grads_fn()

        # Synchronise loss and gradient if needed
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
        c_1: float,
        c_2: float,
        max_iter: int = 20,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform the zoom phase of the strong Wolfe line search,
        narrowing the interval [alpha_lo, alpha_hi] until conditions are met
        or maximum iterations exceeded. Returns (alpha_j, loss, gradient).
        """
        tol_tensor = self.state["tol_tensor"]
        for _ in range(max_iter):
            # Interpolate new trial alpha within [alpha_lo, alpha_hi]
            if _ == 0:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
            else:
                denom = 2 * (phi_hi - phi_lo - dphi_lo * (alpha_hi - alpha_lo))
                cond = abs(denom) > self.tol
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

            # Evaluate function, gradient, and derivative at alpha_j
            phi_j, grad_j, dphi_j = self._evaluate_function_and_gradient(
                wk, p, alpha_j, closure
            )

            # Check Armijo condition or if phi_j >= phi_lo
            armijo = phi_j > phi_0 + c_1 * alpha_j * dphi_0
            if armijo or (phi_j >= phi_lo):
                alpha_hi, phi_hi = alpha_j, phi_j
            else:
                # Check strong Wolfe curvature condition
                strong_wolfe = abs(dphi_j) <= -c_2 * dphi_0
                if strong_wolfe:
                    return alpha_j, phi_j, grad_j
                switch = dphi_j * (alpha_hi - alpha_lo) >= 0
                if switch:
                    alpha_hi, phi_hi = alpha_lo, phi_lo
                # Update lower bound
                alpha_lo, phi_lo, dphi_lo = alpha_j, phi_j, dphi_j

            # Terminate if interval is sufficiently small
            if abs(alpha_hi - alpha_lo) < tol_tensor:
                break

        # If maximum iterations exceeded, return best lower bound
        return alpha_lo, phi_lo, self._flat_grads_fn()

    def _strong_wolfe_line_search(
        self,
        wk: Tensor,
        p: Tensor,
        phi_0: Tensor,
        dphi_0: Tensor,
        closure: Callable,
        alpha_0: float = 1.0,
        c_1: float = 1e-4,
        c_2: float = 0.9,
        alpha_max: float = 10.0,
        max_iter: int = 10,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Conduct strong Wolfe line search to find step size alpha
        that satisfies both Armijo and curvature conditions.
        Returns chosen alpha, new loss, and new gradient.
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

            cond1 = phi_i > phi_0 + c_1 * alpha_i * dphi_0
            cond2 = (i > 0) and (phi_i >= phi_prev)
            # If either condition triggers, enter zoom phase
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
                    c_1,
                    c_2,
                )

            # If curvature condition satisfied, accept alpha_i
            if abs(dphi_i) <= -c_2 * dphi_0:
                return alpha_i, phi_i, grad_i

            # If derivative becomes positive, perform zoom with swapped bounds
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
                    c_1,
                    c_2,
                )

            # Update previous iterate values and double alpha_i (capped at alpha_max)
            alpha_prev, phi_prev, dphi_prev = alpha_i, phi_i, dphi_i
            alpha_i = torch.min(2 * alpha_i, torch.tensor(alpha_max, device=wk.device))
            if alpha_i >= alpha_max:
                break

        # Fallback: return best known step (alpha_prev)
        phi_fin, grad_fin, _ = self._evaluate_function_and_gradient(
            wk, p, alpha_prev.item(), closure
        )
        return alpha_prev, phi_fin, grad_fin

    def _backtracking_line_search(
        self,
        wk: Tensor,
        p: Tensor,
        phi_0: Tensor,
        dphi_0: Tensor,
        closure: Callable,
        alpha_0: float = 1.0,
        c_1: float = 1e-4,
        c_2: float = 0.9,
        max_iter: int = 10,
    ) -> Tuple[float, float, Tensor]:
        """
        Simple backtracking line search to satisfy Armijo and curvature.
        Returns step size alpha, new loss, and new gradient.
        """
        alpha = alpha_0
        tol_tensor = self.state["tol_tensor"]
        for _ in range(max_iter):
            phi_alpha, grad_alpha, dphi_alpha = self._evaluate_function_and_gradient(
                wk, p, alpha, closure
            )
            armijo_ok = phi_alpha <= phi_0 + c_1 * alpha * dphi_0
            curvature_ok = abs(dphi_alpha) <= c_2 * abs(dphi_0)
            if armijo_ok and curvature_ok:
                return alpha, phi_alpha, grad_alpha
            alpha *= 0.5
            if alpha < tol_tensor.item():
                break
        # If line search fails, return zero step
        return 0.0, phi_0, self._flat_grads_fn()

    def step(self, closure: Callable[[], Tensor], **_) -> Tuple[float, float]:
        """
        Perform a single optimisation step.
        Returns tuple (new_loss, flat_gradient).
        """
        # Evaluate or retrieve precomputed loss and gradient
        loss = _["loss"] if "loss" in _ else closure(compute_grad=True)
        g = _["grad"] if "grad" in _ else self._flat_grads_fn()

        gn = torch.norm(g, p=self.norm_type)
        if self.sync and self.world_size > 1:
            loss = self._avg_scalar(loss)
            dist.broadcast(g, src=0)
            gn = self._avg_scalar(gn)
        # If gradient norm below tolerance, skip update
        if gn <= self.tol:
            return loss, g

        # Flatten current parameters and preserve a copy for updates
        wk_flat = self._flat_params_fn()  # self._flatten_params()
        wk = wk_flat.clone().detach()
        st = self.state
        sec = self.second_order

        # Update LSR1 memory if second-order is enabled and previous gradient exists
        if sec and st["prev_grad"] is not None:
            sk = wk - st["old_wk"]
            yk = g - st["prev_grad"]
            if sk.norm() > self.tol and yk.norm() > self.tol:
                self.hess.update_memory(sk, yk)
        st["old_wk"], st["prev_grad"] = wk.clone().detach(), g.clone().detach()

        # Solve trust-region subproblem: second-order if memory available, otherwise first-order
        if sec and len(self.hess._S) > 0:
            p_star, pred = solve_tr_second_order(
                g, gn, self.delta, self.hess, self.obs, self.tol
            )
        else:
            p_star, pred = solve_tr_first_order(g, gn, self.delta, self.tol)

        # Momentum-like update for vk term, bounding to trust-region radius
        vk = st["flat_vk"]
        vk.mul_(self.mu).add_(wk - st["old_wk"])
        vk_norm_sq = vk.dot(vk)
        if vk_norm_sq > 0.0:
            vk_norm = math.sqrt(float(vk_norm_sq))
            scale = min(1.0, self.delta / vk_norm)
            vk.mul_(scale)
        # Combine p_star and vk, then bound combined step to trust-region radius
        p_comb = p_star + vk
        p_comb_norm_sq = p_comb.dot(p_comb)
        if p_comb_norm_sq > 0.0:
            p_comb_norm = math.sqrt(float(p_comb_norm_sq))
            scale = min(1.0, self.delta / p_comb_norm)
            p_comb.mul_(scale)
        st["flat_vk"] = vk.clone().detach()  # Store updated vk for next iteration

        # Prepare line search with initial loss and directional derivative
        phi_0 = loss
        dphi_0 = g.dot(p_comb)

        # Perform strong Wolfe line search to compute step length alpha
        alpha, new_loss, new_g = self._strong_wolfe_line_search(
            wk,
            p_comb,
            phi_0,
            dphi_0,
            closure,
            alpha_0=self.defaults["lr"],
            c_1=self.c_1,
            c_2=self.c_2,
            alpha_max=self.alpha_max,
            max_iter=self.max_wolfe_iters,
        )

        # Apply final parameter update: w_new = wk + alpha * p_comb
        p_step = alpha * p_comb
        self._unflatten_update(wk + p_step)

        # Compute trust-region ratio ρ = (f(new) − f(old)) / predicted
        rho = (new_loss - loss) / pred if (alpha > 0 and pred < 0) else 0.0
        s_norm_sq = p_step.dot(p_step)
        s_norm = math.sqrt(float(s_norm_sq))
        delta_old = self.delta

        # Adjust trust-region radius based on ρ and step norm
        if rho < self.tau_2:
            # Shrink trust region if poor agreement
            delta_new = min(
                self.nu_1 * delta_old,
                self.nu_2 * s_norm**2,
            )
        elif rho >= self.tau_3 and s_norm >= self.nu_3 * delta_old:
            # Expand trust region if very successful
            delta_new = self.nu_4 * delta_old
        else:
            delta_new = delta_old

        # Clip trust-region radius between [min_delta, max_delta]
        self.delta = max(
            self.min_delta,
            min(delta_new, self.max_delta),
        )

        return new_loss, new_g
