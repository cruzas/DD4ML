import copy
import math

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from dd4ml.utility import (
    clone_model,
    flatten_params,
    get_trust_region_params,
    get_local_trust_region_params,
    get_lssr1_trust_region_params,
    get_lssr1_local_trust_region_params,
    get_state_dict,
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
            config.global_optimizer = LSSR1_TR 
            config.global_optimizer_args = get_lssr1_trust_region_params(config) 
        else:
            config.global_optimizer = TR
            config.global_optimizer_args = get_trust_region_params(config)
            
        if config.local_second_order:
            config.local_optimizer = LSSR1_TR 
            config.local_optimizer_args = get_lssr1_local_trust_region_params(config) 
        else:
            config.local_optimizer = TR
            config.local_optimizer_args = get_local_trust_region_params(config)

        print(f"APTS_D global optimizer: {config.global_optimizer.__name__}; local optimizer: {config.local_optimizer.__name__}")
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
        min_delta=None,
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
            diff_tol=1e-8,  # Default tolerance for diff comparison
            delta=float(delta),
            min_delta=float(min_delta) if min_delta is not None else 1e-3,
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
        self.loc_model = clone_model(model)
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
        self.grad_evals = torch.tensor(0.0, device=self.device)

        # Buffers for flattened parameters -------------------------------- #
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
        self._local_flat_buffer = torch.empty_like(sample_flat)

        # Inner optimisers ------------------------------------------------- #
        self.glob_optim = global_opt(self.model.parameters(), **global_opt_params)
        self.loc_optim = local_opt(
            self.loc_model.parameters(), **local_opt_params
        )
        self.delta = self.glob_optim.defaults["delta"]  # will be kept in sync
        self.batch = -1

    def update_pytorch_lr(self) -> None:
        """Update the learning rate in PyTorch's param_groups."""
        for g in self.param_groups:
            g["lr"] = self.delta

    # --------------------------------------------------------------------- #
    # Closure helpers
    # --------------------------------------------------------------------- #
    def non_foc_loc_closure(self, compute_grad: bool = False):
        self.loc_optim.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def foc_loc_closure(self, compute_grad: bool = False):
        self.loc_optim.zero_grad()
        loc_loss = self.non_foc_loc_closure(compute_grad)

        # Flattened parameters (global vs. local) -------------------------- #
        global_flat = flatten_params(self.model, self._flat_params_buffer)
        local_flat = flatten_params(self.loc_model, self._local_flat_buffer)
        diff = local_flat - global_flat
        if not torch.all(torch.abs(diff) < self.defaults["diff_tol"]):
            if self.resid.dim() == 0 and self.resid.item() == 0:
                self.resid = torch.zeros_like(diff)
            loc_loss = loc_loss + (self.resid @ diff)
        return loc_loss

    def glob_closure(self, compute_grad: bool = False):
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP takes care of gradient averaging
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    # --------------------------------------------------------------------- #
    # One optimization step
    # --------------------------------------------------------------------- #
    def step(self, inputs, labels):
        self.batch += 1
        # Short-hand hyper-parameters ------------------------------------- #
        hp = self.defaults
        norm_type = hp["norm_type"]

        # ------------------------------------------------------------------ #
        # Book-keeping and initial evaluations
        # ------------------------------------------------------------------ #
        self.grad_evals.zero_()
        loc_grad_evals = torch.tensor(0.0, device=self.device)
        
        self.inputs, self.labels = inputs, labels
        loc_closure = (
            self.foc_loc_closure if hp["foc"] else self.non_foc_loc_closure
        )
        with torch.no_grad():
            init_glob_flat = flatten_params(self.model, self._flat_params_buffer).clone()
        
        init_glob_loss = self.glob_closure(compute_grad=True)
        with torch.no_grad():
            self.grad_evals += 1
        init_loc_loss = loc_closure(compute_grad=True)
        loc_grad_evals += 1
        
        with torch.no_grad():
            init_glob_grad = parameters_to_vector(
                [p.grad for p in self.model.parameters()]
            ).detach()
            loc_grad = (
                parameters_to_vector([p.grad for p in self.loc_model.parameters()])
                .detach()
                .clone()
            )
        self.resid = init_glob_grad - loc_grad

        # ------------------------------------------------------------------ #
        # Local steps
        # ------------------------------------------------------------------ #
        loc_loss = init_loc_loss
        for _ in range(hp["max_local_iters"]):
            loc_loss, loc_grad = self.loc_optim.step(
                closure=loc_closure,
                precomp_loss=loc_loss,
                precomp_grad=loc_grad,
            )
            loc_grad_evals += 1
            loc_grad_norm = loc_grad.norm(p=norm_type)
            if self.nr_models > 1:
                dist.all_reduce(loc_grad_norm, op=dist.ReduceOp.MAX)
            if loc_grad_norm.item() <= self.loc_optim.defaults["tol"]:
                break
            
        if self.nr_models > 1:
            dist.all_reduce(loc_grad_evals, op=dist.ReduceOp.SUM)
            loc_grad_evals /= self.nr_models
        self.grad_evals += loc_grad_evals

        curr_flat = flatten_params(self.loc_model, self._local_flat_buffer)
        step_vec = curr_flat - init_glob_flat

        # ------------------------------------------------------------------ #
        # Step correction / aggregation
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # Ensure tensor type for arithmetic
            if isinstance(loc_loss, (int, float)):
                loc_loss = torch.tensor(loc_loss, device=self.device)
            elif torch.is_tensor(loc_loss):
                loc_loss = loc_loss.to(self.device)
            else:
                raise TypeError(f"Unexpected type for loc_loss: {type(loc_loss)}. Expected int, float, or torch.Tensor.")

            loc_red = init_loc_loss - loc_loss.detach()
            if self.nr_models > 1:
                # Sum step vector and local reduction across all local models
                dist.all_reduce(step_vec, op=dist.ReduceOp.SUM)
                dist.all_reduce(loc_red, op=dist.ReduceOp.SUM)
                loc_red /= self.nr_models
                if norm_type == math.inf:
                    step_vec /= self.nr_models
            # Set the model parameters to the new values for trial evaluation
            restore_params(self.model, init_glob_flat + step_vec)

        # ------------------------------------------------------------------ #
        # Global acceptance test
        # ------------------------------------------------------------------ #
        if loc_red == 0:
            accept_ratio = float('inf')  # Assign a large value to indicate no reduction
        else:
            trial_loss = self.glob_closure(compute_grad=True)
            trial_grad = parameters_to_vector(
                [p.grad for p in self.model.parameters()]
            ).detach()
            accept_ratio = (init_glob_loss - trial_loss) / loc_red
        with torch.no_grad():
            if accept_ratio < self.defaults["nu_dec"]:
                self.delta = max(self.delta * self.defaults["dec_factor"], self.defaults["min_delta"])
                self.update_pytorch_lr()
                restore_params(self.model, init_glob_flat)
                new_loss = init_glob_loss
                new_grad = init_glob_grad
            else:
                if accept_ratio > self.defaults["nu_inc"]:
                    self.delta = min(
                        self.delta * self.defaults["inc_factor"],
                        self.defaults["max_delta"],
                    )
                    self.update_pytorch_lr() 
                new_loss = trial_loss
                new_grad = trial_grad

            self.glob_optim.defaults["delta"] = self.delta

        # ------------------------------------------------------------------ #
        # Optional global pass
        # ------------------------------------------------------------------ #
        if hp["global_pass"]:
            # Perform global optimization steps using the global optimizer.
            # The `step` method is expected to minimize the loss function
            # provided by the `closure` and optionally use precomputed loss
            # or gradient values for efficiency.
            for _ in range(hp["max_global_iters"]):
                extra_args = {"precomp_loss": new_loss}
                extra_args["precomp_grad"] = new_grad
                
                # Perform the global step
                new_loss, new_grad = self.glob_optim.step(
                    closure=self.glob_closure, **extra_args
                )
            self.grad_evals += 1

        # ------------------------------------------------------------------ #
        with torch.no_grad():
            self.delta = self.glob_optim.defaults["delta"]
            self.update_pytorch_lr()
            
            self.loc_optim.delta = self.glob_optim.defaults["delta"]
            if norm_type != math.inf and self.nr_models > 1:
                self.loc_optim.defaults["delta"] /= self.nr_models
            # Only update local_model if its state differs from model's state
            if any(
                not torch.equal(p1, p2)
                for p1, p2 in zip(self.loc_model.parameters(), self.model.parameters())
            ):
                self.loc_model.load_state_dict(get_state_dict(self.model))

        return new_loss
