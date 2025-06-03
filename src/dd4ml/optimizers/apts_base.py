# common_imports.py (or at top of your module)
import copy
import math
import random

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
    mark_trainable,
    restore_params,
    trainable_parameters_to_vector,
    get_apts_params,
    ensure_tensor
)
from .tr import TR
from .lssr1_tr import LSSR1_TR

class APTS_Base(Optimizer):
    @staticmethod
    def setup_APTS_args(config):
        """
        Configure global and local optimiser classes and arguments
        based on whether second-order methods are required.
        """
        config.global_optimizer, config.global_optimizer_args = (
            (LSSR1_TR, get_lssr1_trust_region_params(config))
            if config.global_second_order
            else (TR, get_trust_region_params(config))
        )
        config.local_optimizer, config.local_optimizer_args = (
            (LSSR1_TR, get_lssr1_local_trust_region_params(config))
            if config.local_second_order
            else (TR, get_local_trust_region_params(config))
        )

        config.apts_params = get_apts_params(config)
        return config

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
        norm_type=2,
        max_local_iters=3,
        max_global_iters=3,
        tol=1e-6,
    ):
        # Register hyper‐parameters through `defaults` ----------------------- #
        defaults = dict(
            global_pass=bool(global_pass),       # whether to run extra global steps
            norm_type=norm_type,                 # norm type for gradient
            max_local_iters=max_local_iters,     # max iterations for local pass
            max_global_iters=max_global_iters,   # max iterations for global pass
            tol=float(tol),                      # tolerance for trust‐region test
            diff_tol=1e-8,                       # tolerance for difference check
            delta=float(delta),                  # initial trust‐region radius
            min_delta=float(min_delta) if min_delta is not None else 1e-3,
            lr=float(delta),                     # keep lr in sync with delta
            nu_dec=float(nu_dec) if nu_dec is not None else 0.25,
            nu_inc=float(nu_inc) if nu_inc is not None else 0.75,
            inc_factor=float(inc_factor) if inc_factor is not None else 1.2,
            dec_factor=float(dec_factor) if dec_factor is not None else 0.9,
            max_delta=float(max_delta) if max_delta is not None else 2.0,
        )
        super().__init__(params, defaults)

        # Common state --------------------------------------------------------- #
        self.model = model
        self.loc_model = None # To be set in subclasses
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

        # Buffers for flattened parameters: pre‐allocate to avoid reallocations
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
        self._loc_flat_buffer = torch.empty_like(sample_flat)

        # Instantiate global optimiser (trust‐region or LSSR1_TR)
        self.glob_optim = global_opt(
            self.model.parameters(), **global_opt_params
        )
    
        # Keep delta in sync with underlying global optimiser
        self.delta = self.glob_optim.defaults["delta"]
        
        # Track number of gradient evaluations as a simple Python float
        self.grad_evals = 0.0
        self.local_grad_evals = 0.0
        
    def update_pytorch_lr(self) -> None:
        """
        Synchronise PyTorch param_groups' learning rate to current delta.
        """
        for g in self.param_groups:
            g["lr"] = self.defaults["delta"]
        
    def non_foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure when no first-order correction is used.
        Computes and optionally backpropagates the local loss.
        """
        self.loc_optim.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure with first-order correction. Adds residual term
        to local loss if global vs. local parameters diverge.
        """
        self.loc_optim.zero_grad()
        loc_loss = self.non_foc_loc_closure(compute_grad)

        # Flatten global and local parameters into pre-allocated buffers
        global_flat = flatten_params(self.model, self._flat_params_buffer)
        local_flat = flatten_params(self.loc_model, self._loc_flat_buffer)
        diff = local_flat - global_flat

        # If difference above tolerance, add residual inner-product term
        if not torch.all(torch.abs(diff) < self.defaults["diff_tol"]):
            if self.resid.dim() == 0 and self.resid.item() == 0:
                self.resid = torch.zeros_like(diff)
            loc_loss = loc_loss + (self.resid @ diff)
        return loc_loss

    def glob_closure(self, compute_grad: bool = False):
        """
        Global closure: zeroes gradients, computes loss on the global model,
        optionally backpropagates, and reduces loss across processes if needed.
        """
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP will handle gradient averaging
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss
    
    def loc_steps(self, loc_loss, loc_grad, loc_closure):
        for _ in range(self.defaults["max_local_iters"]):
            # Perform a local trust-region (or LSSR1) step with precomputed values
            loc_loss, loc_grad = self.loc_optim.step(
                closure=loc_closure,
                loss=loc_loss,
                grad=loc_grad,
            )
            self.loc_grad_evals += 1

            # Compute gradient norm for stopping criterion
            loc_grad_norm = loc_grad.norm(p=self.defaults["norm_type"])
            if self.nr_models > 1:
                # Synchronise max norm across processes
                dist.all_reduce(loc_grad_norm, op=dist.ReduceOp.MAX)
            # Stop local iterations if gradient norm below tolerance
            if loc_grad_norm <= self.loc_optim.defaults["tol"]:
                break
        return loc_loss, loc_grad
    
    def glob_steps(self, loss, grad):
        for _ in range(self.defaults["max_global_iters"]):
            loss, grad = self.glob_optim.step(
                closure=self.glob_closure, loss=loss, grad=grad
            )
            # Stop local iterations if gradient norm below tolerance
            if grad.norm(p=self.defaults["norm_type"]) <= self.glob_optim.defaults["tol"]:
                break
        return loss, grad
    
    @torch.no_grad()
    def glob_grad_to_vector(self):
        """
        Converts the global model's gradients to a flattened vector.
        Returns a tensor containing the gradients of the global model parameters.
        """
        return parameters_to_vector([p.grad for p in self.model.parameters()])
        
    @torch.no_grad()
    def loc_grad_to_vector(self):
        """
        Converts the local model's gradients to a flattened vector.
        Returns a tensor containing the gradients of the local model parameters.
        """
        return parameters_to_vector([p.grad for p in self.loc_model.parameters()])
    
    def apts_tr_control(
        self,
        step: torch.Tensor,
        pred: torch.Tensor):
        """
        Trust-region control logic to decide whether to accept the step.
        Returns True if the step is accepted, False otherwise.
        """
        # Apply proposed step: update global model parameters temporarily
        with torch.no_grad():
            restore_params(self.model, self.init_glob_flat + step)
        
        # Compute trial loss and gradient
        trial_loss = self.glob_closure(compute_grad=True)
        trial_grad = self.glob_grad_to_vector()
        
        # Compute acceptance ratio; if no local reduction, force rejection
        if pred < self.defaults["tol"]:
            accept_ratio = float('inf')
        else:
            accept_ratio = (self.init_glob_loss - trial_loss) / pred # actual / predicted
        
        with torch.no_grad():
            if pred < self.defaults["tol"] or accept_ratio < self.defaults["nu_dec"]:
                # Reject step: shrink trust region, restore original params
                self.defaults["delta"] = max(self.defaults["delta"] * self.defaults["dec_factor"], self.defaults["min_delta"])
                self.update_pytorch_lr()
                restore_params(self.model, self.init_glob_flat)
                loss = self.init_glob_loss
                grad = self.init_glob_grad
            else:
                # Accept step: possibly enlarge trust region
                if accept_ratio > self.defaults["nu_inc"]:
                    self.defaults["delta"] = min(
                        self.defaults["delta"] * self.defaults["inc_factor"],
                        self.defaults["max_delta"],
                    )
                    self.update_pytorch_lr()
                loss = trial_loss
                grad = trial_grad
        return loss, grad
                
