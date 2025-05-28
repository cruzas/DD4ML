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
    get_lssr1_trust_region_params,
    get_state_dict,
    get_trust_region_params,
    restore_params,
    get_apts_params
)

from .lssr1_tr import LSSR1_TR
from .tr import TR


class APTS_D(Optimizer):
    # --------------------------------------------------------------------- #
    # Static helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def setup_APTS_args(config):
        if config.global_second_order:
            config.global_optimizer = TR #LSSR1_TR
            config.global_optimizer_args = get_trust_region_params(config) #get_lssr1_trust_region_params(config)
        else:
            config.global_optimizer = TR
            config.global_optimizer_args = get_trust_region_params(config)
        if config.local_second_order:
            config.local_optimizer = TR# LSSR1_TR
            config.local_optimizer_args = get_trust_region_params(config) #get_lssr1_trust_region_params(config)
        else:
            config.subdomain_optimizer = TR
            config.subdomain_optimizer_args = get_local_trust_region_params(config)

        print(f"APTS_D global optimizer: {TR.__name__}; local optimizer: {TR.__name__}")
        config.apts_params = get_apts_params(config)
        return config

    # --------------------------------------------------------------------- #
    # Initialisation
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        params,
        model=None,
        delta=None,
        max_delta=None,
        nu_dec=None,
        nu_inc=None,
        inc_factor=None,
        dec_factor=None,
        criterion=None,
        device=None,
        nr_models=None,
        global_opt=None,
        global_opt_params=None,
        local_opt=None,
        local_opt_params=None,
        *,
        global_pass=True,
        foc=True,
        correct_step=True,
        norm_type=2,
        max_local_iters=3,
        max_global_iters=3,
        tol=1e-6,
    ):
        # Register hyper-parameters through ``defaults`` ------------------- #
        defaults = dict(
            global_pass=bool(global_pass),
            foc=bool(foc),
            correct_step=bool(correct_step),
            norm_type=norm_type,
            max_local_iters=max_local_iters,
            max_global_iters=max_global_iters,
            tol=float(tol),
            delta=float(delta),
            lr=float(delta),
            nu_dec=float(nu_dec) if nu_dec is not None else 0.25,
            nu_inc=float(nu_inc) if nu_inc is not None else 0.75,
            inc_factor=float(inc_factor) if inc_factor is not None else 1.2,
            dec_factor=float(dec_factor) if dec_factor is not None else 0.9,
            max_delta=float(max_delta) if max_delta is not None else 2.0,
        )
        super().__init__(params, defaults)

        # Basic state ------------------------------------------------------ #
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
        self.grad_evals_counter = torch.zeros(1, device=self.device)

        # Buffers for flattened parameters -------------------------------- #
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
        self._local_flat_buffer = torch.empty_like(sample_flat)

        # Inner optimisers ------------------------------------------------- #
        self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
        self.local_optimizer = local_opt(
            self.local_model.parameters(), **local_opt_params
        )
        self.delta = self.global_optimizer.defaults["delta"]  # will be kept in sync

    def update_pytorch_lr(self) -> None:
        """Update the learning rate in PyTorch's param_groups."""
        for g in self.param_groups:
            g["lr"] = self.delta

    # --------------------------------------------------------------------- #
    # Closure helpers
    # --------------------------------------------------------------------- #
    def non_foc_local_closure(self, compute_grad: bool = False):
        self.local_optimizer.zero_grad()
        loss = self.criterion(self.local_model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def foc_local_closure(self, compute_grad: bool = False):
        self.local_optimizer.zero_grad()
        local_loss = self.non_foc_local_closure(compute_grad)

        # Flattened parameters (global vs. local) -------------------------- #
        global_flat = flatten_params(self.model, self._flat_params_buffer)
        local_flat = flatten_params(self.local_model, self._local_flat_buffer)
        diff = local_flat - global_flat
        if not torch.all(torch.abs(diff) < 1e-8):
            if self.residual.dim() == 0 and self.residual.item() == 0:
                self.residual = torch.zeros_like(diff)
            local_loss = local_loss + (self.residual @ diff)
        return local_loss

    def global_closure(self, compute_grad: bool = False):
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP takes care of gradient averaging
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    # --------------------------------------------------------------------- #
    # One optimisation step
    # --------------------------------------------------------------------- #
    def step(self, inputs, labels):
        # Short-hand hyper-parameters ------------------------------------- #
        hp = self.defaults
        norm_type = hp["norm_type"]

        # ------------------------------------------------------------------ #
        # Book-keeping and initial evaluations
        # ------------------------------------------------------------------ #
        self.grad_evals_counter.zero_()
        self.inputs, self.labels = inputs, labels
        local_closure = (
            self.foc_local_closure if hp["foc"] else self.non_foc_local_closure
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

        # ------------------------------------------------------------------ #
        # Local steps
        # ------------------------------------------------------------------ #
        for _ in range(hp["max_local_iters"]):
            local_loss, local_grad = self.local_optimizer.step(
                closure=local_closure,
                precom_loss=initial_local_loss,
                precom_grad=local_grad,
            )
            total_local_grad_evals_counter += 1
            local_grad_norm = local_grad.norm(p=norm_type).item()
            if local_grad_norm <= self.local_optimizer.defaults["tol"]:
                break

        if self.nr_models > 1:
            dist.all_reduce(total_local_grad_evals_counter, op=dist.ReduceOp.SUM)
            total_local_grad_evals_counter /= self.nr_models
        self.grad_evals_counter += total_local_grad_evals_counter

        current_flat = flatten_params(self.local_model, self._local_flat_buffer)
        step_vec = current_flat - initial_flat

        # ------------------------------------------------------------------ #
        # Step correction / aggregation
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # Ensure tensor type for arithmetic
            if not torch.is_tensor(local_loss):
                local_loss = torch.tensor(local_loss, device=self.device)
            else:
                local_loss = local_loss.to(self.device)

            local_reduction = initial_local_loss - local_loss.detach()
            if self.nr_models > 1:
                dist.all_reduce(step_vec, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_reduction, op=dist.ReduceOp.SUM)
                if norm_type == math.inf:
                    step_vec /= self.nr_models

            if hp["correct_step"]:
                corrected_step = fix_aggregated_local_steps_pnorm(
                    aggregated_step=step_vec, global_grad=global_grad, p=norm_type
                )
                restore_params(self.model, initial_flat + corrected_step)
            else:
                restore_params(self.model, initial_flat + step_vec)

        # ------------------------------------------------------------------ #
        # Global acceptance test
        # ------------------------------------------------------------------ #
        trial_loss = self.global_closure()
        with torch.no_grad():
            acceptance_ratio = (initial_global_loss - trial_loss) / local_reduction

            if acceptance_ratio < self.defaults["nu_dec"]:
                self.delta = max(self.delta * self.defaults["dec_factor"], self.defaults["tol"])
                self.update_pytorch_lr()
                
                restore_params(self.model, initial_flat)
                new_loss = initial_global_loss
            else:
                if acceptance_ratio > self.defaults["nu_inc"]:
                    self.delta = min(
                        self.delta * self.defaults["inc_factor"],
                        self.defaults["max_delta"],
                    )
                    self.update_pytorch_lr()
                new_loss = trial_loss

            self.global_optimizer.defaults["delta"] = self.delta

        # ------------------------------------------------------------------ #
        # Optional global pass
        # ------------------------------------------------------------------ #
        if hp["global_pass"]:
            for _ in range(hp["max_global_iters"]):
                extra_args = {"precomp_loss": new_loss}
                if new_loss == initial_global_loss:
                    extra_args["precomp_grad"] = global_grad

                # Perform the global step
                new_loss, new_grad = self.global_optimizer.step(
                    closure=self.global_closure, **extra_args
                )
            self.grad_evals_counter += 1

        # ------------------------------------------------------------------ #
        # House-keeping
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            self.delta = self.global_optimizer.defaults["delta"]
            self.update_pytorch_lr()
            
            self.local_optimizer.delta = self.global_optimizer.defaults["delta"]
            if norm_type != math.inf:
                self.local_optimizer.defaults["delta"] /= self.nr_models
            self.local_model.load_state_dict(get_state_dict(self.model))

        return new_loss
