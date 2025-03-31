import copy
import math

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from .trust_region_legacy_code import TrustRegion  # Explicit import
from .utils import (
    get_local_trust_region_params,
    get_state_dict,
    get_trust_region_params,
)


def fix_aggregated_local_steps_pnorm(
    aggregated_step, global_grad, p=2.0, tol=1e-6, max_iter=50
):
    """
    Projects aggregated_step onto the direction of global_grad under the p-norm.

    Cases:
      - p = 2: Uses closed-form projection.
      - p = 1: Uses weighted median of ratios a_i/g_i.
      - p = float('inf'): Minimizes f(alpha)=max_i|a_i - alpha*g_i| via ternary search.
      - Otherwise (p > 1, p ≠ 2): Finds optimal alpha via derivative-based bisection.
    """
    # Return zero if global_grad is zero.
    if torch.norm(global_grad) == 0:
        return torch.zeros_like(global_grad)

    # Special case: p = 2
    if abs(p - 2.0) < tol:
        norm_global_sq = torch.dot(global_grad, global_grad)
        alpha = (
            torch.dot(aggregated_step, global_grad) / norm_global_sq
            if norm_global_sq > 0
            else 0.0
        )
        return alpha * global_grad

    # Special case: p = 1 (minimize L1 norm)
    elif abs(p - 1.0) < tol:
        valid = global_grad != 0
        if valid.sum() == 0:
            return torch.zeros_like(global_grad)
        # Compute ratios and weights for indices with nonzero global_grad.
        ratios = (aggregated_step[valid] / global_grad[valid]).detach().cpu().numpy()
        weights = torch.abs(global_grad[valid]).detach().cpu().numpy()
        sorted_indices = ratios.argsort()
        ratios_sorted = ratios[sorted_indices]
        weights_sorted = weights[sorted_indices]
        total_weight = weights_sorted.sum()
        cumulative = 0.0
        for r, w in zip(ratios_sorted, weights_sorted):
            cumulative += w
            if cumulative >= total_weight / 2:
                alpha = r
                break
        return alpha * global_grad

    # Special case: p = infinity (minimize L-inf norm)
    elif p == float("inf"):
        valid = global_grad != 0
        if valid.sum() == 0:
            return torch.zeros_like(global_grad)
        # Use ratios from valid indices to bracket the optimum.
        ratios = (aggregated_step[valid] / global_grad[valid]).detach().cpu().numpy()
        L, R = ratios.min(), ratios.max()

        def f(alpha):
            diff = aggregated_step - alpha * global_grad
            return torch.max(torch.abs(diff)).item()

        for _ in range(max_iter):
            m1 = L + (R - L) / 3
            m2 = R - (R - L) / 3
            if f(m1) > f(m2):
                L = m1
            else:
                R = m2
        alpha = (L + R) / 2
        return alpha * global_grad

    # General case: p > 1 (and not 1,2,inf)
    else:

        def derivative(alpha):
            diff = aggregated_step - alpha * global_grad
            # Derivative of f(alpha)=||aggregated_step-alpha*global_grad||_p^p (up to constant factor)
            return torch.sum(
                global_grad * torch.sign(diff) * torch.abs(diff) ** (p - 1)
            )

        d0 = derivative(0.0)
        if abs(d0) < tol:
            return torch.zeros_like(global_grad)

        if d0 > 0:
            alpha_high = 0.0
            alpha_low = -1.0
            while derivative(alpha_low) > 0 and abs(alpha_low) < 1e6:
                alpha_low *= 2.0
        else:
            alpha_low = 0.0
            alpha_high = 1.0
            while derivative(alpha_high) < 0 and abs(alpha_high) < 1e6:
                alpha_high *= 2.0

        for _ in range(max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2.0
            d_mid = derivative(alpha_mid)
            if abs(d_mid) < tol:
                return alpha_mid * global_grad
            if d_mid > 0:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
        return ((alpha_low + alpha_high) / 2.0) * global_grad


def flatten_params(model, out=None):
    if out is None:
        return parameters_to_vector(model.parameters())
    # Write flattened parameters into a preallocated tensor.
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        out[offset : offset + numel].copy_(param.data.view(-1))
        offset += numel
    return out


def restore_params(model, flat_params):
    vector_to_parameters(flat_params, model.parameters())


def clone_model(model):
    base_model = model.module if hasattr(model, "module") else model
    config_copy = copy.deepcopy(base_model.config)
    config_copy.model_type = None
    new_model = type(base_model)(config_copy)
    new_model.load_state_dict(get_state_dict(base_model))
    return new_model.to(model.device)


class APTS_D(Optimizer):
    @staticmethod
    def setup_APTS_args(config):
        config.global_optimizer = TrustRegion
        config.global_optimizer_args = get_trust_region_params(config)

        config.subdomain_optimizer = TrustRegion
        config.subdomain_optimizer_args = get_local_trust_region_params(config)

        return config

    def __init__(
        self,
        params,
        model=None,
        criterion=None,
        device=None,
        nr_models=None,
        global_opt=None,
        global_opt_params=None,
        local_opt=None,
        local_opt_params=None,
        global_pass=True,
        foc=True,
        correct_step=True,
        norm_type=2,
    ):
        super().__init__(params, {})
        self.model = model
        self.local_model = clone_model(model)
        self.nr_models = nr_models
        self.device = (
            device
            if device is not None
            else (
                f"cuda:{torch.cuda.current_device()}"
                if getattr(self, "backend", "cuda") != "gloo"
                else "cpu"
            )
        )
        self.criterion = criterion
        self.global_pass = bool(global_pass)
        self.foc = bool(foc)
        self.grad_evals_counter = torch.zeros(1, device=self.device)

        # Preallocate two buffers for flattening: one for global and one for local.
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
        self._local_flat_buffer = torch.empty_like(sample_flat)

        if "TrustRegion" in str(global_opt):
            self.global_optimizer = global_opt(self.model, **global_opt_params)
            self.local_optimizer = local_opt(self.local_model, **local_opt_params)
        else:
            self.global_optimizer = global_opt(
                self.model.parameters(), **global_opt_params
            )
            self.local_optimizer = local_opt(
                self.local_model.parameters(), **local_opt_params
            )
        self.lr = self.global_optimizer.lr
        self.correct_step = correct_step
        self.norm_type = norm_type

    def non_foc_local_closure(self, compute_grad=False):
        self.local_optimizer.zero_grad()
        outputs = self.local_model(self.inputs)
        loss = self.criterion(outputs, self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def foc_local_closure(self, compute_grad=False):
        self.local_optimizer.zero_grad()
        local_loss = self.non_foc_local_closure(compute_grad)
        # Compute global and local flattened parameters using separate buffers.
        global_flat = flatten_params(self.model, self._flat_params_buffer)
        local_flat = flatten_params(self.local_model, self._local_flat_buffer)
        diff = local_flat - global_flat
        if not torch.all(torch.abs(diff) < 1e-8):
            if self.residual.dim() == 0 and self.residual.item() == 0:
                self.residual = torch.zeros_like(diff)
            local_loss = local_loss + (self.residual @ diff)
        return local_loss

    def global_closure(self, compute_grad=False):
        self.zero_grad()
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP handles gradient averaging
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    def step(self, inputs, labels):
        self.grad_evals_counter.zero_()
        self.inputs, self.labels = inputs, labels
        local_closure = (
            self.foc_local_closure if self.foc else self.non_foc_local_closure
        )

        with torch.no_grad():
            initial_flat = flatten_params(self.model, self._flat_params_buffer).clone()

        initial_global_loss = self.global_closure()
        self.grad_evals_counter += 1

        total_local_grad_evals_counter = torch.tensor(0.0, device=self.device)
        initial_local_loss = local_closure()
        total_local_grad_evals_counter += 1

        global_grad = parameters_to_vector(
            [p.grad for p in self.model.parameters()]
        ).detach()
        local_grad = (
            parameters_to_vector([p.grad for p in self.local_model.parameters()])
            .detach()
            .clone()
        )
        self.residual = global_grad - local_grad

        # Local steps
        step_vec = None
        for local_iter in range(self.local_optimizer.max_iter):
            loss_arg = initial_local_loss if local_iter == 0 else None
            grad_arg = local_grad if local_iter == 0 else None
            local_loss = self.local_optimizer.step(
                closure=local_closure, old_loss=loss_arg, grad=grad_arg
            )

            with torch.no_grad():
                if local_iter > 0:
                    total_local_grad_evals_counter += 1

                current_flat = flatten_params(self.local_model, self._local_flat_buffer)
                step_vec = current_flat - initial_flat  # local step vector

                max_lr_reached = (
                    torch.norm(step_vec, p=self.norm_type)
                    >= self.local_optimizer.max_lr
                )
                max_iter_reached = (
                    self.local_optimizer.local_iter >= self.local_optimizer.max_iter
                )
                if max_lr_reached or max_iter_reached:
                    break

        if self.nr_models > 1:
            dist.all_reduce(total_local_grad_evals_counter, op=dist.ReduceOp.SUM)
            total_local_grad_evals_counter /= self.nr_models

        self.grad_evals_counter += total_local_grad_evals_counter

        with torch.no_grad():
            # If local loss is not a torch.Tensor, it is a scalar.
            if not torch.is_tensor(local_loss):
                local_loss = torch.tensor(local_loss, device=self.device)
            else:
                local_loss = local_loss.to(self.device)

            local_reduction = initial_local_loss - local_loss.detach()
            if self.nr_models > 1:
                dist.all_reduce(step_vec, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_reduction, op=dist.ReduceOp.SUM)

                if self.norm_type == math.inf:
                    step_vec /= self.nr_models

            if self.correct_step:
                corrected_step = fix_aggregated_local_steps_pnorm(
                    aggregated_step=step_vec, global_grad=global_grad, p=self.norm_type
                )
                restore_params(self.model, initial_flat + corrected_step)
            else:
                restore_params(self.model, initial_flat + step_vec)

        trial_loss = self.global_closure()

        with torch.no_grad():
            acceptance_ratio = (initial_global_loss - trial_loss) / local_reduction

            if acceptance_ratio < self.global_optimizer.nu_1:
                self.lr = max(
                    self.lr * self.global_optimizer.dec_factor,
                    self.global_optimizer.min_lr,
                )
                restore_params(self.model, initial_flat)
                new_loss = initial_global_loss
                grad_arg = global_grad
            elif acceptance_ratio > self.global_optimizer.nu_2:
                self.lr = min(
                    self.lr * self.global_optimizer.inc_factor,
                    self.global_optimizer.max_lr,
                )
                new_loss = trial_loss
                grad_arg = None
            else:
                new_loss = trial_loss
                grad_arg = None

            self.global_optimizer.lr = self.lr

        if self.global_pass:
            for _ in range(self.global_optimizer.max_iter):
                new_loss = self.global_optimizer.step(
                    closure=self.global_closure, old_loss=new_loss, grad=grad_arg
                )
                grad_arg = None
                self.grad_evals_counter += 1

        with torch.no_grad():
            self.lr = self.global_optimizer.lr
            self.local_optimizer.lr = self.lr / self.nr_models
            self.local_model.load_state_dict(get_state_dict(self.model))

        return new_loss
