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
        """
        Configure global and local optimiser classes and arguments
        based on whether second‐order methods are required.
        """
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

        config.apts_params = get_apts_params(config)
        return config

    # --------------------------------------------------------------------- #
    # Initialization
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
        norm_type=2,
        max_local_iters=3,
        max_global_iters=3,
        tol=1e-6,
    ):
        # Register hyper‐parameters through `defaults` ------------------- #
        defaults = dict(
            global_pass=bool(global_pass),       # whether to run extra global steps
            foc=bool(foc),                       # first‐order correction toggle
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

        # Basic state ------------------------------------------------------ #
        self.model = model
        # Clone model for local updates; avoids overwriting global params
        self.loc_model = clone_model(model)
        self.nr_models = nr_models
        # Determine device: use provided or infer current CUDA device if not Gloo
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

        # Track number of gradient evaluations as a simple Python float
        self.grad_evals = 0.0

        # Buffers for flattened parameters: pre‐allocate to avoid repeated allocations
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
        self._local_flat_buffer = torch.empty_like(sample_flat)

        # Instantiate inner optimisers for global and local passes
        self.glob_optim = global_opt(self.model.parameters(), **global_opt_params)
        self.loc_optim = local_opt(
            self.loc_model.parameters(), **local_opt_params
        )
        # Keep delta in sync with underlying optimisers
        self.delta = self.glob_optim.defaults["delta"]
        self.batch = -1  # batch counter

    def update_pytorch_lr(self) -> None:
        """
        Synchronise PyTorch param_groups' learning rate to current delta.
        """
        for g in self.param_groups:
            g["lr"] = self.delta

    # --------------------------------------------------------------------- #
    # Closure helpers
    # --------------------------------------------------------------------- #
    def non_foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure when no first‐order correction is used.
        Computes and optionally backpropagates the local loss.
        """
        self.loc_optim.zero_grad()
        loss = self.criterion(self.loc_model(self.inputs), self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def foc_loc_closure(self, compute_grad: bool = False):
        """
        Local closure with first‐order correction. Adds residual term
        to local loss if global vs. local parameters diverge.
        """
        self.loc_optim.zero_grad()
        loc_loss = self.non_foc_loc_closure(compute_grad)

        # Flatten global and local parameters into pre‐allocated buffers
        global_flat = flatten_params(self.model, self._flat_params_buffer)
        local_flat = flatten_params(self.loc_model, self._local_flat_buffer)
        diff = local_flat - global_flat

        # If difference above tolerance, add residual inner‐product term
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

    # --------------------------------------------------------------------- #
    # One optimisation step
    # --------------------------------------------------------------------- #
    def step(self, inputs, labels):
        """
        Performs one APTS_D step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.
        """
        self.batch += 1
        hp = self.defaults  # shorthand for hyper‐parameters
        norm_type = hp["norm_type"]

        # ------------------------------------------------------------------ #
        # Book‐keeping and initial evaluations
        # ------------------------------------------------------------------ #
        # Reset gradient evaluation counter (as Python float)
        self.grad_evals = 0.0
        loc_grad_evals = 0  # track local grad evals as Python int

        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        # Choose local closure based on first‐order correction flag
        loc_closure = (
            self.foc_loc_closure if hp["foc"] else self.non_foc_loc_closure
        )

        # Save initial global parameters (flattened, cloned to avoid in‐place)
        with torch.no_grad():
            init_glob_flat = flatten_params(self.model, self._flat_params_buffer).clone()

        # Compute initial global loss and gradient
        init_glob_loss = self.glob_closure(compute_grad=True)
        # Count one global gradient eval
        self.grad_evals += 1.0

        # Compute initial local loss and gradient
        init_loc_loss = loc_closure(compute_grad=True)
        loc_grad_evals += 1

        # Store initial gradients (flattened)
        with torch.no_grad():
            init_glob_grad = parameters_to_vector(
                [p.grad for p in self.model.parameters()]
            ).detach()
            loc_grad = (
                parameters_to_vector([p.grad for p in self.loc_model.parameters()])
                .detach()
                .clone()
            )
        # Calculate residual between global and local gradients
        self.resid = init_glob_grad - loc_grad

        # ------------------------------------------------------------------ #
        # Local steps
        # ------------------------------------------------------------------ #
        loc_loss = init_loc_loss
        for _ in range(hp["max_local_iters"]):
            # Perform a local trust‐region (or LSSR1) step with precomputed values
            loc_loss, loc_grad = self.loc_optim.step(
                closure=loc_closure,
                precomp_loss=loc_loss,
                precomp_grad=loc_grad,
            )
            loc_grad_evals += 1

            # Compute gradient norm for stopping criterion
            loc_grad_norm = loc_grad.norm(p=norm_type)
            if self.nr_models > 1:
                # Synchronise max norm across processes
                dist.all_reduce(loc_grad_norm, op=dist.ReduceOp.MAX)
            # Stop local iterations if gradient norm below tolerance
            if loc_grad_norm <= self.loc_optim.defaults["tol"]:
                break

        # Account for local gradient evaluations across all models
        self.grad_evals += loc_grad_evals * self.nr_models

        # Compute step vector: current local params minus initial global params
        curr_flat = flatten_params(self.loc_model, self._local_flat_buffer)
        step_vec = curr_flat - init_glob_flat

        # ------------------------------------------------------------------ #
        # Step correction / aggregation
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # Ensure loc_loss is a tensor to allow arithmetic
            if isinstance(loc_loss, (int, float)):
                loc_loss = torch.tensor(loc_loss, device=self.device)
            elif torch.is_tensor(loc_loss):
                loc_loss = loc_loss.to(self.device)
            else:
                raise TypeError(f"Unexpected type for loc_loss: {type(loc_loss)}. Expected int, float, or torch.Tensor.")

            # Reduction in local loss
            loc_red = init_loc_loss - loc_loss.detach()

            if self.nr_models > 1:
                # Coalesce step_vec (sized [P]) and loc_red (sized [1]) into one tensor
                coalesced = torch.cat([
                    step_vec.view(-1),
                    loc_red.view(1)
                ])
                # Single all_reduce on that combined tensor
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)

                # Split back into step_vec and loc_red
                numel = step_vec.numel()
                step_vec.copy_(coalesced[:numel].view_as(step_vec))
                loc_red = coalesced[numel].unsqueeze(0)

                if norm_type == math.inf:
                    step_vec /= self.nr_models
                loc_red /= self.nr_models

            # Apply proposed step: update global model parameters temporarily
            restore_params(self.model, init_glob_flat + step_vec)

        # ------------------------------------------------------------------ #
        # Global acceptance test
        # ------------------------------------------------------------------ #
        trial_loss = self.glob_closure(compute_grad=True)
        trial_grad = parameters_to_vector(
            [p.grad for p in self.model.parameters()]
        ).detach()

        # Compute acceptance ratio; if no local reduction, force rejection
        if loc_red < self.defaults["tol"]:
            accept_ratio = float('inf')
        else:
            accept_ratio = (init_glob_loss - trial_loss) / loc_red

        with torch.no_grad():
            if loc_red < self.defaults["tol"] or accept_ratio < self.defaults["nu_dec"]:
                # Reject step: shrink trust region, restore original params
                self.delta = max(self.delta * self.defaults["dec_factor"], self.defaults["min_delta"])
                self.update_pytorch_lr()
                restore_params(self.model, init_glob_flat)
                new_loss = init_glob_loss
                new_grad = init_glob_grad
            else:
                # Accept step: possibly enlarge trust region
                if accept_ratio > self.defaults["nu_inc"]:
                    self.delta = min(
                        self.delta * self.defaults["inc_factor"],
                        self.defaults["max_delta"],
                    )
                    self.update_pytorch_lr()
                new_loss = trial_loss
                new_grad = trial_grad

            # Keep global optimiser's delta in sync
            self.glob_optim.defaults["delta"] = self.delta

        # ------------------------------------------------------------------ #
        # Optional global pass
        # ------------------------------------------------------------------ #
        if hp["global_pass"]:
            # Perform additional global optimisation steps
            for _ in range(hp["max_global_iters"]):
                extra_args = {"precomp_loss": new_loss}
                extra_args["precomp_grad"] = new_grad

                new_loss, new_grad = self.glob_optim.step(
                    closure=self.glob_closure, **extra_args
                )

                # Stop if global gradient norm is sufficiently small
                new_grad_norm = new_grad.norm(p=norm_type)
                if new_grad_norm <= self.glob_optim.defaults["tol"]:
                    break

        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # Sync final delta and update local optimiser accordingly
            self.delta = self.glob_optim.defaults["delta"]
            self.update_pytorch_lr()

            self.loc_optim.delta = self.glob_optim.defaults["delta"]
            if norm_type != math.inf and self.nr_models > 1:
                self.loc_optim.defaults["delta"] /= self.nr_models

            # Ensure local model matches global model for next iteration
            self.loc_model.load_state_dict(get_state_dict(self.model))

        return new_loss
