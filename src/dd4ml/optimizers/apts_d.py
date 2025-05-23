import copy
import math

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from dd4ml.utility import (
    clone_model,
    fix_aggregated_local_steps_pnorm,
    flatten_params,
    get_local_trust_region_params,
    get_state_dict,
    get_trust_region_params,
    restore_params,
)

from .trust_region_ema import TrustRegionEMA
from .trust_region_first_order import TrustRegionFirstOrder  # Explicit import
from .trust_region_second_order import TrustRegionSecondOrder  # Explicit import


class APTS_D(Optimizer):
    @staticmethod
    def setup_APTS_args(config):
        if config.ema and config.global_second_order:
            raise ValueError(
                "APTS_D global optimizer does not support second-order optimizers with EMA."
            )
        glob_optim_class = TrustRegionFirstOrder
        if config.ema:
            glob_optim_class = TrustRegionEMA
        elif config.global_second_order:
            glob_optim_class = TrustRegionSecondOrder

        config.global_optimizer = glob_optim_class
        config.global_optimizer_args = get_trust_region_params(config)

        loc_optim_class = TrustRegionFirstOrder
        if config.local_second_order:
            loc_optim_class = TrustRegionSecondOrder

        config.subdomain_optimizer = loc_optim_class
        config.subdomain_optimizer_args = get_local_trust_region_params(config)

        print(
            f"APTS_D global optimizer: {glob_optim_class.__name__}; local optimizer: {loc_optim_class.__name__}"
        )
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
        local_loss = self.local_optimizer.step(
            closure=local_closure, old_loss=initial_local_loss, grad=local_grad
        )
        total_local_grad_evals_counter += self.local_optimizer.local_iter
        if self.nr_models > 1:
            dist.all_reduce(total_local_grad_evals_counter, op=dist.ReduceOp.SUM)
            total_local_grad_evals_counter /= self.nr_models
        self.grad_evals_counter += total_local_grad_evals_counter

        current_flat = flatten_params(self.local_model, self._local_flat_buffer)
        step_vec = current_flat - initial_flat
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
            new_loss = self.global_optimizer.step(
                closure=self.global_closure, old_loss=new_loss, grad=grad_arg
            )
            self.grad_evals_counter += 1

        with torch.no_grad():
            self.lr = self.global_optimizer.lr
            self.local_optimizer.lr = self.global_optimizer.lr
            if self.norm_type != math.inf:
                self.local_optimizer.lr /= self.nr_models
            self.local_model.load_state_dict(get_state_dict(self.model))

        return new_loss
