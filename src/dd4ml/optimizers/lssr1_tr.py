from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_lssr1_tr_hparams
from dd4ml.utility.optimizer_utils import solve_tr_first_order, solve_tr_second_order

from .lsr1 import LSR1


class LSSR1_TR(Optimizer):
    __name__ = "LSSR1_TR"

    @staticmethod
    def setup_LSSR1_TR_hparams(cfg):
        """
        Setup hyperparameters for the LSSR1_TR optimizer based on the provided config.
        This function extracts relevant parameters and returns them as a dictionary.
        """
        for k, v in get_lssr1_tr_hparams(cfg).items():
            setattr(cfg, k, v)
        return cfg

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
        dogleg: bool = False,  # only used if second_order is True
        mem_length: int = 10,
        max_wolfe_iters: int = 5,
        max_zoom_iters: int = 5,
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
        paper_tr_update: bool = True,
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
        self.dogleg = dogleg  # dogleg is only used in second-order mode
        if self.dogleg and not self.second_order:
            raise ValueError("Dogleg is only applicable in second-order mode")
        self.mem_length = mem_length
        self.max_wolfe_iters = max_wolfe_iters
        self.max_zoom_iters = max_zoom_iters
        self.mu = mu
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu_3 = nu_3
        self.nu_4 = nu_4
        self.tol = float(tol)
        self.norm_type = 2
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

        # Instantiate OBS solver for trust-region subproblem
        self.obs = OBS()
        # Instantiate limited-memory SR1 Hessian approximation
        self.hess = LSR1(
            gamma=self.gamma,
            memory_length=self.mem_length,
            device=device,
            dtype=dtype,
            tol=self.tol,
        )

        # Flags for distributed synchronization
        self.sync = bool(sync)
        self.paper_tr_update = bool(paper_tr_update)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Caching for efficiency
        self._hess_memory_size_cache = 0
        self._precomputed_for_size = -1

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

        buf = value.detach().to(torch.float64)
        dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            buf /= self.world_size
        dist.broadcast(buf, src=0)
        return buf.to(value.dtype)

    def _avg_scalar_batch(self, values: Tensor) -> Tensor:
        """
        Perform reduce-average on multiple scalar tensors across ranks, then broadcast.
        If `sync` is False or single process, return values unchanged.
        """
        if not (self.sync and self.world_size > 1):
            return values

        buf = values.detach().to(torch.float64)
        dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            buf /= self.world_size
        dist.broadcast(buf, src=0)
        return buf.to(values.dtype)

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
        return buf.clone()

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
        return buf.clone()

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
        Synchronize scalars and gradient if in distributed mode.
        """
        # Update parameters to the trial point
        self._unflatten_update(wk + alpha * p)

        # Compute loss and gradients via closure
        loss = closure(compute_grad=True)
        grad = self._flat_grads_fn()

        # Compute directional derivative along p
        deriv = grad.dot(p)

        # Synchronize loss and gradient if needed
        if self.sync and self.world_size > 1:
            scalars = torch.tensor([loss, deriv], dtype=loss.dtype, device=loss.device)
            scalars_synced = self._avg_scalar_batch(scalars)
            loss, deriv = scalars_synced[0], scalars_synced[1]
            dist.broadcast(grad, src=0)

        return loss, grad, deriv

    @staticmethod
    def _cubic_interpolate(
        x1: Tensor,
        f1: Tensor,
        g1: Tensor,
        x2: Tensor,
        f2: Tensor,
        g2: Tensor,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        Full cubic interpolation between (x1,f1,g1) and (x2,f2,g2),
        clamped to bounds if provided.
        """
        if bounds is not None:
            xmin, xmax = bounds
        else:
            xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)

        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_sq = d1.pow(2) - g1 * g2
        if d2_sq >= 0:
            d2 = d2_sq.sqrt()
            if x1 <= x2:
                pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            else:
                pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            return torch.clamp(pos, xmin, xmax)
        # fallback to bisection
        return (xmin + xmax) / 2

    def _zoom(
        self,
        wk: Tensor,
        p: Tensor,
        alpha_lo: Tensor,
        alpha_hi: Tensor,
        phi_lo: Tensor,
        phi_hi: Tensor,
        dphi_lo: Tensor,
        dphi_hi: Tensor,
        phi_0: Tensor,
        dphi_0: Tensor,
        closure: Callable,
        c_1: float,
        c_2: float,
        max_iter: int = 5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Zoom phase using full cubic interpolation between (alpha_lo, phi_lo, dphi_lo)
        and (alpha_hi, phi_hi, dphi_hi), clamped to a safe interval.
        Returns (alpha_best, phi_best, grad_best).
        """
        grad_j_last = None
        for i in range(max_iter):
            if i == 0:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
            else:
                safe_low = alpha_lo + 0.1 * (alpha_hi - alpha_lo)
                safe_high = alpha_hi - 0.1 * (alpha_hi - alpha_lo)
                alpha_j = self._cubic_interpolate(
                    alpha_lo,
                    phi_lo,
                    dphi_lo,
                    alpha_hi,
                    phi_hi,
                    dphi_hi,
                    bounds=(safe_low, safe_high),
                )

            phi_j, grad_j, dphi_j = self._evaluate_function_and_gradient(
                wk, p, alpha_j, closure
            )
            grad_j_last = grad_j

            # Check Armijo or bracket lower
            if phi_j > (phi_0 + c_1 * alpha_j * dphi_0) or phi_j >= phi_lo:
                alpha_hi, phi_hi, dphi_hi = alpha_j, phi_j, dphi_j
            else:
                # Check curvature condition
                if abs(dphi_j) <= -c_2 * dphi_0:
                    return alpha_j, phi_j, grad_j
                # If derivative signs imply we passed the minimiser
                if dphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi, phi_hi, dphi_hi = alpha_lo, phi_lo, dphi_lo
                # Move lower bound up
                alpha_lo, phi_lo, dphi_lo = alpha_j, phi_j, dphi_j

            # Terminate if interval is sufficiently small
            if (alpha_hi - alpha_lo) * p.abs().max() < self.tol:
                break

        # Fallback: return best lower bound
        return alpha_lo, phi_lo, grad_j_last

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
        max_iter: int = 5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Conduct strong Wolfe line search with cubic-zoom.
        Returns (alpha, new_loss, new_grad).
        """
        alpha_prev = alpha_0
        phi_prev = phi_0
        dphi_prev = dphi_0

        # initial trial
        alpha_i = 0.5 * alpha_max

        for i in range(max_iter):
            phi_i, grad_i, dphi_i = self._evaluate_function_and_gradient(
                wk, p, alpha_i, closure
            )

            armijo = phi_i > (phi_0 + c_1 * alpha_i * dphi_0)
            bracket_cond = (i > 1) and (phi_i >= phi_prev)

            if armijo or bracket_cond:
                return self._zoom(
                    wk=wk,
                    p=p,
                    alpha_lo=alpha_prev,
                    alpha_hi=alpha_i,
                    phi_lo=phi_prev,
                    phi_hi=phi_i,
                    dphi_lo=dphi_prev,
                    dphi_hi=dphi_i,
                    phi_0=phi_0,
                    dphi_0=dphi_0,
                    closure=closure,
                    c_1=c_1,
                    c_2=c_2,
                    max_iter=self.max_zoom_iters,
                )

            # curvature condition
            if abs(dphi_i) <= -c_2 * dphi_0:
                return alpha_i, phi_i, grad_i

            # derivative positive => reverse bracket
            if dphi_i >= 0:
                return self._zoom(
                    wk=wk,
                    p=p,
                    alpha_lo=alpha_i,
                    alpha_hi=alpha_prev,
                    phi_lo=phi_i,
                    phi_hi=phi_prev,
                    dphi_lo=dphi_i,
                    dphi_hi=dphi_prev,
                    phi_0=phi_0,
                    dphi_0=dphi_0,
                    closure=closure,
                    c_1=c_1,
                    c_2=c_2,
                    max_iter=self.max_zoom_iters,
                )

            # update and expand
            alpha_prev, phi_prev, dphi_prev, grad_prev = (
                alpha_i,
                phi_i,
                dphi_i,
                grad_i,
            )
            alpha_i = min(1.25 * alpha_i, alpha_max)
            if alpha_i >= alpha_max:
                break

        # fallback
        return alpha_prev, phi_prev, grad_prev

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
        max_iter: int = 1,
    ) -> Tuple[float, float, Tensor]:
        """
        Simple backtracking line search to satisfy Armijo and curvature.
        Returns step size alpha, new loss, and new gradient.
        """
        alpha = alpha_0
        for _ in range(max_iter):
            phi_alpha, grad_alpha, dphi_alpha = self._evaluate_function_and_gradient(
                wk, p, alpha, closure
            )
            armijo_ok = phi_alpha <= phi_0 + c_1 * alpha * dphi_0
            curvature_ok = abs(dphi_alpha) <= c_2 * abs(dphi_0)
            if armijo_ok and curvature_ok:
                return alpha, phi_alpha, grad_alpha
            alpha *= 0.5
            if alpha < self.tol:
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
        gn = torch.norm(g)

        if self.sync and self.world_size > 1:
            scalars = torch.tensor([loss, gn], dtype=loss.dtype, device=loss.device)
            scalars_synced = self._avg_scalar_batch(scalars)
            loss, gn = scalars_synced[0], scalars_synced[1]
            dist.broadcast(g, src=0)

        if gn <= self.tol:
            return loss, g

        # Flatten current parameters and preserve a copy for updates
        wk = self._flat_params_fn()
        st = self.state

        # Update LSR1 memory if second-order is enabled and previous gradient exists
        hess_memory_updated = False
        if self.second_order and st["prev_grad"] is not None:
            sk = wk - st["old_wk"]
            yk = g - st["prev_grad"]
            if sk.norm() > self.tol and yk.norm() > self.tol:
                self.hess.update_memory(sk, yk)  # Also takes care of updating gamma
                self.gamma = self.hess.gamma
                hess_memory_updated = True

        st["old_wk"], st["prev_grad"] = wk.clone(), g.clone()

        # Only precompute if Hessian memory changed
        current_memory_size = len(self.hess._S)
        if (self.second_order and current_memory_size > 0 and 
            (hess_memory_updated or self._precomputed_for_size != current_memory_size)):
            self.hess.precompute()
            self._precomputed_for_size = current_memory_size

        # Solve trust-region subproblem: second-order if memory available, otherwise first-order
        if self.second_order and current_memory_size > 0:
            # pred_red = -(g*p + 0.5*p*B*p)
            p_star, pred_red = solve_tr_second_order(
                gradient=g,
                grad_norm=gn,
                trust_radius=self.delta,
                lsr1_hessian=self.hess,
                obs_solver=self.obs,
                tol=self.tol,
                dogleg=self.dogleg,
            )
        else:
            # pred_red = -g*p
            p_star, pred_red = solve_tr_first_order(g, gn, self.delta, self.tol)

        # Momentum-like update for vk term, bounding to trust-region radius
        vk = st["flat_vk"]
        vk.mul_(self.mu).add_(wk - st["old_wk"])
        
        # Cache norm computations
        vk_norm_sq = vk.dot(vk)
        if vk_norm_sq > self.tol:
            vk_norm = math.sqrt(float(vk_norm_sq))
            scale = min(1.0, self.delta / vk_norm)
            vk.mul_(scale)
            
        # Combine p_star and vk, then bound combined step to trust-region radius
        p_comb = p_star + vk
        p_comb_norm_sq = p_comb.dot(p_comb)
        if p_comb_norm_sq > self.tol:
            p_comb_norm = math.sqrt(float(p_comb_norm_sq))
            scale = min(1.0, self.delta / p_comb_norm)
            p_comb.mul_(scale)
            # Update cached norm after scaling
            p_comb_norm_sq = p_comb.dot(p_comb)
            
        st["flat_vk"] = vk.clone()
        
        # Cache frequently used values for line search
        st["_p_comb_norm_sq"] = p_comb_norm_sq

        # Prepare line search with initial loss and directional derivative
        phi_0 = loss
        dphi_0 = g.dot(p_comb)

        # Assumption 3 in paper
        if dphi_0.item() > 0:
            p_comb.mul_(-1)
            pred_red *= -1
            dphi_0.mul_(-1)

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

        # Compute trust-region ratio ρ = (f(new) − f(old)) / pred_redicted
        if abs(float(pred_red)) < self.tol:
            rho = float("inf")
        else:
            rho = (loss - new_loss) / pred_red if (alpha > 0 and pred_red < 0) else 0.0
            
        # Use cached norm if available and step is just scaled version
        if alpha == 1.0 and "_p_comb_norm_sq" in st:
            s_norm_sq = st["_p_comb_norm_sq"]
        else:
            s_norm_sq = p_step.dot(p_step)
        s_norm = math.sqrt(float(s_norm_sq))

        # Adjust trust-region radius based on ρ and step norm
        if self.paper_tr_update:
            delta_old = self.delta
            if rho < self.tau_2:
                # Shrink trust region if poor agreement
                delta_new = min(
                    self.nu_1 * delta_old,
                    self.nu_2 * s_norm,
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
        else:
            if rho > self.tau_2 and rho > self.tau_3 and s_norm >= 0.9 * self.delta:
                # Accept the step and increase delta
                self.delta = min(
                    self.max_delta,
                    self.nu_4 * self.delta,
                )
            else:
                # Reject the step and decrease delta
                self.delta = max(
                    self.min_delta,
                    self.nu_3 * self.delta,
                )

        return new_loss, new_g.clone()
