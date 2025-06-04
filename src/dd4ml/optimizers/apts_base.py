import copy
import math
import random
import time

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from dd4ml.utility import (
    Timer,
    clone_model,
    dprint,
    ensure_tensor,
    flatten_params,
    get_apts_params,
    get_device,
    get_loc_tr_hparams,
    get_lssr1_loc_tr_hparams,
    get_lssr1_tr_hparams,
    get_state_dict,
    get_tr_hparams,
    mark_trainable,
    restore_params,
    trainable_params_to_vector,
)

from .lssr1_tr import LSSR1_TR
from .tr import TR


class APTS_Base(Optimizer):
    __name__ = "APTS_Base"

    @staticmethod
    def setup_APTS_hparams(config):
        """
        Configure global and local optimizer classes and arguments
        based on whether second-order methods are required.
        """
        config.glob_opt, config.glob_opt_hparams = (
            (LSSR1_TR, get_lssr1_tr_hparams(config))
            if config.glob_second_order
            else (TR, get_tr_hparams(config))
        )
        config.loc_opt, config.loc_opt_hparams = (
            (LSSR1_TR, get_lssr1_loc_tr_hparams(config))
            if config.loc_second_order
            else (TR, get_loc_tr_hparams(config))
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
        # Only 'lr' remains in defaults
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

        # Instantiate global optimizer (trust-region or LSSR1_TR)
        self.model = model
        self.glob_opt = glob_opt(self.model.parameters(), **glob_opt_hparams)

        # To be set in subclasses
        self.loc_model = None
        self.loc_opt = None
        self.loc_closure = None

        # Keep delta in sync with underlying global optimizer
        self.delta = self.glob_opt.delta

        # Buffers for flattened parameters: pre-allocate to avoid reallocations
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)  # for global model
        self._loc_flat_buffer = torch.empty_like(sample_flat)  # for local models

        # Track number of gradient evaluations as simple Python floats
        self.grad_evals = 0.0
        self.loc_grad_evals = 0.0

    def zero_timers(self):
        self.timings = {key: 0 for key in self.timings}
        if hasattr(self.loc_opt, "zero_timers"):
            self.loc_opt.zero_timers()

    def get_timings(self):
        timings = self.timings.copy()
        if "tradam" in str(self.loc_opt).lower():
            timings.update(self.loc_opt.get_timings())
        return timings

    def display_avg_timers(self):
        timings = self.get_timings()
        timings.pop("precond", None)
        total_time = sum(timings.values())
        headers = ["Timer", "Time (s)", "Percentage (%)"]
        rows = [
            [k, f"{v:.4f}", f"{(v / total_time) * 100:.2f}%"]
            for k, v in sorted(timings.items())
        ]
        col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(3)]
        row_fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        table = [row_fmt.format(*headers), "-" * sum(col_widths)]
        table.extend(row_fmt.format(*row) for row in rows)
        table_str = "\n".join(table)
        print(table_str)
        return table_str

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
        return parameters_to_vector(
            [p.grad for p in self.loc_model.parameters()]
        ).detach()

    @torch.no_grad()
    def glob_params_to_vector(self):
        """
        Converts the global model's parameters to a flattened vector.
        Returns a tensor containing the parameters of the global model.
        """
        return flatten_params(self.model, self._flat_params_buffer).detach()

    @torch.no_grad()
    def loc_params_to_vector(self):
        """
        Converts the local model's parameters to a flattened vector.
        Returns a tensor containing the parameters of the local model.
        """
        return flatten_params(self.loc_model, self._loc_flat_buffer).detach()

    @torch.no_grad()
    def ensure_step_within_tr(self, step):
        """
        Ensure the step is within the trust region delta.
        If the step norm exceeds delta, scale it down.
        """
        step_norm = step.norm(p=self.norm_type)
        if step_norm > self.delta:
            dprint(
                f"Warning: step norm {step_norm:.6e} exceeds delta {self.delta:.4f}. Scaling down step."
            )
            step = (self.delta / step_norm) * step
        return step

    @torch.no_grad()
    def sync_glob_to_loc(self):
        self.delta = self.glob_opt.delta
        self.update_pytorch_lr()

        self.loc_opt.delta = self.glob_opt.delta
        if self.norm_type != math.inf and self.nr_models > 1:
            self.loc_opt.delta /= self.nr_models
        self.loc_opt.update_pytorch_lr()

        # Ensure local model matches global model for next iteration
        self.loc_model.load_state_dict(get_state_dict(self.model))

    def non_foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure when no first-order correction is used.
        Computes and optionally backpropagates the local loss.
        """
        self.loc_opt.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
            self.loc_grad_evals += 1  # Count local gradient evaluation
        return loss

    def foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure with first-order correction. Adds residual term
        to local loss if global vs. local parameters diverge.
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

    def glob_closure(self, compute_grad: bool = False):
        """
        Global closure: zeroes gradients, computes loss on the global model,
        optionally backpropagates, and reduces loss across processes if needed.
        """
        self.zero_grad()
        loss = self.criterion(self.model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # DDP will handle gradient averaging
            self.grad_evals += 1.0  # Count global gradient evaluation
        if self.nr_models > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.nr_models
        return loss

    def _step_loop(self, optim, max_iters, loss, grad, closure=None):
        for _ in range(max_iters):
            # Perform an optimization step (trust-region or LSSR1) with precomputed values
            if closure is not None:
                loss, grad = optim.step(closure=closure, loss=loss, grad=grad)
            else:
                raise ValueError(
                    "Closure must be provided for global/local optimization step in APTS. To be modified in future."
                )

            # # Compute gradient norm for stopping criterion
            # grad_norm = grad.norm(p=self.norm_type)
            # if self.nr_models > 1:
            #     # Synchronize max norm across processes
            #     dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)

            # Stop iterations if gradient norm is below tolerance
            if grad_norm <= optim.tol:
                break

        return loss, grad

    def loc_steps(self, loss, grad):
        return self._step_loop(
            optim=self.loc_opt,
            max_iters=self.max_loc_iters,
            loss=loss,
            grad=grad,
            closure=self.loc_closure,
        )

    def glob_steps(self, loss, grad, closure=None):
        return self._step_loop(
            optim=self.glob_opt,
            max_iters=self.max_glob_iters,
            loss=loss,
            grad=grad,
            closure=self.glob_closure if closure is None else closure,
        )

    @torch.no_grad()
    def control_step(self, step: torch.Tensor, pred: torch.Tensor | None = None):
        """
        Unified trust-region control:
          - If `pred` is given, use it directly (pure TR).
          - Otherwise compute a Dogleg prediction:
              • First-order fallback (pred = -gᵀp) when no SR1 memory.
              • Full (gᵀp + 0.5·pᵀBp) if `hess._S` is nonempty.
        """

        # Tentatively apply the step
        restore_params(self.model, self.init_glob_flat + step)

        # Evaluate trial loss & gradient
        trial_loss = self.glob_closure(compute_grad=True)
        trial_grad = self.glob_grad_to_vector()

        # Compute or use supplied `pred`
        if pred is None:
            # Dogleg-style prediction
            g = self.init_glob_grad
            # linear part
            pred_val = -g.dot(step)
            # add quadratic term if SR1 info exists
            if len(self.glob_opt.hess._S) > 0:
                self.glob_opt.hess.precompute()
                Bp = self.glob_opt.hess.B(step)
                pred_val -= 0.5 * step.dot(Bp)
        else:
            pred_val = pred

        # Compute rho = (f(init) − f(trial)) / pred
        if pred_val.abs() < self.tol:
            rho = float("inf")
        else:
            rho = (self.init_glob_loss - trial_loss) / pred_val

        # Accept/reject + adjust trust region
        if pred_val < self.tol or rho < self.nu_dec:
            # Reject: shrink trust region and restore original params
            self.delta = max(self.delta * self.dec_factor, self.min_delta)
            self.update_pytorch_lr()
            restore_params(self.model, self.init_glob_flat)
            return self.init_glob_loss, self.init_glob_grad, self.delta
        else:
            # Accept: possibly enlarge trust region
            if rho > self.nu_inc:
                self.delta = min(self.delta * self.inc_factor, self.max_delta)
                self.update_pytorch_lr()
            return trial_loss, trial_grad, self.delta
