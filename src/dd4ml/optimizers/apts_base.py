import copy
import math
import random
import time

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
from torch.optim.optimizer import Optimizer

from dd4ml.utility import (
    apts_ip_restore_params,
    clone_model,
    dprint,
    ensure_tensor,
    flatten_params,
    get_apts_hparams,
    get_device,
    get_loc_tr_hparams,
    get_loc_tradam_hparams,
    get_lssr1_loc_tr_hparams,
    get_lssr1_tr_hparams,
    get_state_dict,
    get_tr_hparams,
    mark_trainable,
    restore_params,
    trainable_grads_to_vector,
    trainable_params_to_vector,
)

from .asntr import ASNTR
from .lssr1_tr import LSSR1_TR
from .tr import TR
from .tradam import TRAdam


class APTS_Base(Optimizer):
    __name__ = "APTS_Base"

    @staticmethod
    def setup_APTS_hparams(config):
        """
        Configure global and local optimizer classes and arguments
        based on whether second-order methods are required.
        """
        glob_map = {
            "tr": (TR, get_tr_hparams),
            "lssr1_tr": (LSSR1_TR, get_lssr1_tr_hparams),
        }
        loc_map = {
            "tr": (TR, get_loc_tr_hparams),
            "lssr1_tr": (LSSR1_TR, get_lssr1_loc_tr_hparams),
            "tradam": (TRAdam, get_loc_tradam_hparams),
            "sgd": (torch.optim.SGD, lambda _: {"lr": 0.01}),
        }

        if isinstance(config.glob_opt, str):
            key = config.glob_opt.lower()
            try:
                config.glob_opt, hp_fn = glob_map[key]
            except KeyError:
                raise ValueError(f"Unknown glob_opt: {config.glob_opt}")
            config.glob_opt_hparams = hp_fn(config)
        else:
            raise ValueError(
                "glob_opt must be a string specifying the optimizer type, e.g., 'tr' or 'lssr1_tr'."
            )

        loc_value = getattr(config, "loc_opt", None)
        if isinstance(loc_value, str):
            key = loc_value.lower()
            try:
                config.loc_opt, hp_fn = loc_map[key]
            except KeyError:
                raise ValueError(f"Unknown loc_opt: {loc_value}")
            config.loc_opt_hparams = hp_fn(config)
        else:
            raise ValueError(
                "loc_opt must be a string specifying the optimizer type, e.g., 'tr', 'lssr1_tr', or 'sgd'."
            )

        config.apts_params = get_apts_hparams(config)
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
        glob_opt=None,
        glob_opt_hparams=None,
        loc_opt=None,
        loc_opt_hparams=None,
        *,
        glob_pass=True,
        norm_type=None,
        max_loc_iters=None,
        max_glob_iters=None,
        tol=1e-6,
    ):
        # Only ``lr" remains in defaults
        super().__init__(params, {"lr": delta})

        # Assign hyperparameters as attributes
        self.glob_pass = bool(glob_pass)
        self.norm_type = norm_type
        self.max_loc_iters = max_loc_iters
        self.max_glob_iters = max_glob_iters
        self.tol = float(tol)
        self.delta = float(delta)
        self.min_delta = float(min_delta) if min_delta is not None else 1e-3
        self.nu_dec = float(nu_dec) if nu_dec is not None else 0.25
        self.nu_inc = float(nu_inc) if nu_inc is not None else 0.75
        self.inc_factor = float(inc_factor) if inc_factor is not None else 1.2
        self.dec_factor = float(dec_factor) if dec_factor is not None else 0.9
        self.max_delta = float(max_delta) if max_delta is not None else 2.0

        # Common state
        self.nr_models = nr_models  # number of models in the distributed environment
        self.device = get_device(device)
        self.criterion = criterion
        self.dataset_len = 0

        # Instantiate global optimizer (LSSR1_TR, TR, or ASNTR)
        self.model = model
        self.glob_opt = glob_opt(self.model.parameters(), **glob_opt_hparams)

        # To be set in subclasses
        self.loc_model = None
        self.loc_opt = None
        self.loc_closure = None
        self.loc_closure_d = None  # For ASNTR only

        # Keep delta in sync with underlying global optimizer
        self.delta = self.glob_opt.delta

        # Buffers for flattened parameters: pre-allocate to avoid reallocations
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)  # for global model
        self._loc_flat_buffer = torch.empty_like(sample_flat)  # for local models

        # Track number of gradient evaluations as simple Python floats
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0

    def _update_param_group(self):
        for key in self.param_groups[0].keys():
            if key != "params":
                self.param_groups[0][key] = getattr(self, key)

    def _sync_attributes_from_param_group(self):
        for key, value in self.param_groups[0].items():
            if key != "params":
                setattr(self, key, value)

    @torch.no_grad()
    def update_pytorch_lr(self) -> None:
        """
        Synchronize PyTorch param_groups' learning rate to current delta.
        """
        for g in self.param_groups:
            g["lr"] = self.delta

    @torch.no_grad()
    def glob_grad_to_vector(self):
        """
        Converts the global model's gradients to a flattened vector.
        Returns a tensor containing the gradients of the global model parameters.
        """
        return parameters_to_vector([p.grad for p in self.model.parameters()]).detach()

    @torch.no_grad()
    def loc_grad_to_vector(self):
        """
        Converts the local model's gradients to a flattened vector.
        Returns a tensor containing the gradients of the local model parameters.
        """
        return (
            parameters_to_vector([p.grad for p in self.loc_model.parameters()])
            .clone()
            .detach()
        )

    @torch.no_grad()
    def glob_params_to_vector(self):
        """
        Converts the global model's parameters to a flattened vector.
        Returns a tensor containing the parameters of the global model.
        """
        return flatten_params(self.model, self._flat_params_buffer).clone().detach()

    @torch.no_grad()
    def loc_params_to_vector(self):
        """
        Converts the local model's parameters to a flattened vector.
        Returns a tensor containing the parameters of the local model.
        """
        return flatten_params(self.loc_model, self._loc_flat_buffer).clone().detach()

    @torch.no_grad()
    def sync_glob_to_loc(self):
        self.delta = self.glob_opt.delta
        self.update_pytorch_lr()

        self.loc_opt.delta = self.glob_opt.delta
        if self.norm_type == 2 and self.nr_models > 1:
            self.loc_opt.delta /= self.nr_models
        if hasattr(self.loc_opt, "update_pytorch_lr"):
            self.loc_opt.update_pytorch_lr()

        # Ensure local model matches global model for the next iteration
        self.loc_model.load_state_dict(get_state_dict(self.model))

    def non_foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure when no first-order consistency is used.
        Computes and optionally backpropagates the local loss.
        """
        self.loc_opt.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs), self.labels)
        if compute_grad:
            # ``loss`` may be used for multiple backward passes within the
            # trust-region optimisation loop.  Retaining the graph prevents
            # PyTorch from freeing intermediate tensors too early, avoiding
            # ``RuntimeError: Trying to backward through the graph a second
            # time`` when the closure is re-evaluated.
            loss.backward(retain_graph=True)
            self.loc_grad_evals += 1  # Count local gradient evaluation
        return loss

    def foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure with first-order consistency. Adds residual term
        to local loss if global vs local parameters diverge.
        """
        self.loc_opt.zero_grad()
        loss = self.non_foc_loc_closure(compute_grad)

        # Flatten global and local parameters into pre-allocated buffers
        glob_flat = flatten_params(self.model, self._flat_params_buffer)
        loc_flat = flatten_params(self.loc_model, self._loc_flat_buffer)
        diff = loc_flat - glob_flat

        # If difference above tolerance, add residual inner-product term
        if not torch.all(torch.abs(diff) < self.tol):
            if self.resid.dim() == 0 and self.resid.item() == 0:
                self.resid = torch.zeros_like(diff)
            loss = loss + (self.resid @ diff)
        return loss

    def glob_closure_main(self, compute_grad: bool = False):
        """
        Global closure: zeroes gradients, computes loss on the global model,
        optionally backpropagates, and reduces loss across processes if needed.
        """
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP handles gradient averaging
            # Check if model is DDP-wrapped
            if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                # Handle gradient averaging for non-DDP models
                if self.nr_models > 1 and dist.is_available() and dist.is_initialized():
                    # flatten grads, sum across ranks, then average
                    flat_grad = self.glob_grad_to_vector()
                    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
                    flat_grad.div_(self.nr_models)
                    vector_to_parameters(
                        flat_grad, [p.grad for p in self.model.parameters()]
                    )
            self.grad_evals += 1.0  # Count global gradient evaluation
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    # For ASNTR
    def non_foc_loc_closure_d(self, compute_grad: bool = False):
        self.loc_opt.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs_d), self.labels_d)
        if torch.is_grad_enabled() or compute_grad:
            # ``loss`` can be reused by the trust-region solver for multiple
            # gradient evaluations.  Retaining the graph avoids the
            # "backward through the graph a second time" runtime error when
            # the closure is invoked repeatedly within a single optimisation
            # step.
            loss.backward(retain_graph=True)
            self.loc_grad_evals += 1
        return loss

    def foc_loc_closure_d(self, compute_grad: bool = False):
        self.loc_opt.zero_grad()
        loss = self.non_foc_loc_closure_d(compute_grad)

        # Flatten global and local parameters into pre-allocated buffers
        glob_flat = flatten_params(self.model, self._flat_params_buffer)
        loc_flat = flatten_params(self.loc_model, self._loc_flat_buffer)
        diff = loc_flat - glob_flat

        # If difference above tolerance, add residual inner-product term
        if not torch.all(torch.abs(diff) < self.tol):
            if self.resid.dim() == 0 and self.resid.item() == 0:
                self.resid = torch.zeros_like(diff)
            loss = loss + (self.resid @ diff)
        return loss

    def glob_closure_d(self, compute_grad: bool = False):
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs_d), self.labels_d)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP will handle gradient averaging
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    def _step_loop(self, optim, max_iters, loss, grad, closure=None, closure_d=None):
        for _ in range(max_iters):
            # Perform an optimization step (trust-region, LSSR1, or ASNTR) with precomputed values
            if closure is not None:
                # Check if TR or LSSR1_TR optimizer
                if isinstance(optim, (TR, LSSR1_TR)):
                    loss, grad = optim.step(closure=closure, loss=loss, grad=grad)
                else:
                    # For ASNTR, we need to provide closure_d as well
                    loss, grad = optim.step(
                        closure_main=closure,
                        closure_d=closure_d,
                        loss=loss,
                        grad=grad,
                    )
            else:
                raise ValueError(
                    "Closure must be provided for global/local optimization step in APTS. To be modified in future."
                )

            # Stop iterations if gradient norm is below tolerance
            if grad.norm(p=self.norm_type) <= optim.tol:
                break

        return loss, grad

    def loc_steps(self, loss, grad):
        step_loop_args = {
            "optim": self.loc_opt,
            "max_iters": self.max_loc_iters,
            "loss": loss,
            "grad": grad,
            "closure": self.loc_closure,
            "closure_d": self.loc_closure_d,
        }

        loc_loss, loc_grad = self._step_loop(**step_loop_args)
        loc_evals = torch.tensor(self.loc_grad_evals, device=self.device)
        if self.nr_models > 1:
            dist.all_reduce(loc_evals, op=dist.ReduceOp.SUM)
            loc_evals /= self.nr_models
        self.grad_evals += loc_evals.item()
        return loc_loss, loc_grad

    def glob_steps(self, loss, grad, closure=None, closure_d=None):
        step_loop_args = {
            "optim": self.glob_opt,
            "max_iters": self.max_glob_iters,
            "loss": loss,
            "grad": grad,
            "closure": self.glob_closure_main if closure is None else closure,
            "closure_d": self.glob_closure_d if closure_d is None else closure_d,
        }

        return self._step_loop(**step_loop_args)

    def control_step(
        self, step: torch.Tensor, pred: torch.Tensor | None = None, closure=None
    ):
        """
        Unified trust-region control:
          - If "pred" is given, use it directly (pure TR).
          - Otherwise compute prediction:
              + First-order fallback (pred = -gᵀp) when no SR1 memory.
              + Second-order (pred = -(g^T * p + 0.5*p^T * B * p) if SR1 memory available.
        """
        # Determine restore function based on closure presence
        restore_fn = restore_params if closure is None else apts_ip_restore_params
        # Tentatively apply the step
        restore_fn(self.model, self.init_glob_flat + step)

        # Evaluate trial loss & gradient
        trial_loss = (
            self.glob_closure_main(compute_grad=True)
            if closure is None
            else closure(compute_grad=True, zero_grad=True)
        )
        trial_grad = (
            self.glob_grad_to_vector()
            if closure is None
            else self.model.grad(clone=True)
        )

        # Compute or use supplied ``pred"
        if pred is None:
            g = self.init_glob_grad
            # linear part
            pred_red = -g.dot(step)
            # add quadratic term if SR1 info exists
            # Check if hess is an attribute of glob_opt
            if (
                hasattr(self.glob_opt, "hess")
                and self.glob_opt.hess is not None
                and self.glob_opt.hess._S is not None
                and len(self.glob_opt.hess._S) > 0
            ):
                self.glob_opt.hess.precompute()
                Bp = self.glob_opt.hess.B(step)
                pred_red -= 0.5 * step.dot(Bp)
        else:
            pred_red = pred

        # Compute rho = (f(init) − f(trial)) / pred
        if abs(float(pred_red)) < self.tol:
            rho = float("inf")
        else:
            rho = (self.init_glob_loss - trial_loss) / pred_red

        # Accept/reject and adjust trust region radius
        if pred_red < self.tol or rho < self.nu_dec:
            # Reject: restore original params (and possibly shrink the trust region radius)
            self.delta = max(self.delta * self.dec_factor, self.min_delta)
            self.update_pytorch_lr()
            restore_fn(self.model, self.init_glob_flat)
            return self.init_glob_loss, self.init_glob_grad, self.delta
        else:
            # Accept (and possibly enlarge the trust region radius)
            if rho > self.nu_inc:
                self.delta = min(self.delta * self.inc_factor, self.max_delta)
                self.update_pytorch_lr()
            return trial_loss, trial_grad, self.delta
