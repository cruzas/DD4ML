import copy
import math
import random

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from dd4ml.utility import (clone_model, decide_tensor_device, flatten_params,
                           get_local_trust_region_params, get_state_dict,
                           get_trust_region_params, mark_trainable,
                           print_params_norm, restore_params,
                           trainable_parameters_to_vector)

from .trust_region_ema import TrustRegionEMA
from .trust_region_first_order import TrustRegionFirstOrder  # Explicit import
from .trust_region_second_order import \
    TrustRegionSecondOrder  # Explicit import


class APTS_P(Optimizer):
    def __name__(self):
        return "APTS_P"

    @staticmethod
    def setup_APTS_args(config):
        if config.ema and config.global_second_order:
            raise ValueError(
                "APTS_P global optimizer does not support second-order optimizers with EMA."
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
        config.subdomain_optimizer_args = get_trust_region_params(config)

        print(
            f"APTS_P global optimizer: {glob_optim_class.__name__}; local optimizer: {loc_optim_class.__name__}"
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
        correct_step=True,
        norm_type=2,
        dogleg=False,
    ):
        super().__init__(params, {})
        self.model = model
        print("Global model parameters norm...")
        print_params_norm(self.model)

        self.local_model = mark_trainable(clone_model(model))
        print("Local model parameters norm...")
        print_params_norm(self.local_model)

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
        self.global_pass = global_pass

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
        self.dogleg = dogleg

    def local_closure(self, compute_grad=False):
        self.local_optimizer.zero_grad()
        outputs = self.local_model(self.inputs)
        loss = self.criterion(outputs, self.labels)
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()
        return loss

    def global_closure(self, compute_grad=False):
        self.zero_grad()
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.labels)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Average the loss across all processes
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
        if torch.is_grad_enabled() or compute_grad:
            loss.backward()  # no division by N as we want all global models to have the same loss
        return loss

    @torch.no_grad()
    def sync_loc_to_glob(self) -> None:
        """
        Make the global model identical on every rank by merging the
        sub-domain (trainable) updates held in `self.local_model`.

        Assumption: `mark_trainable` assigned *disjoint* index sets, i.e.
        each parameter tensor is owned by exactly one rank.
        """
        # -------- single-process fallback ------------------------------------
        if (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_world_size() == 1
        ):
            with torch.no_grad():
                for pg, pl in zip(
                    self.model.parameters(), self.local_model.parameters()
                ):
                    if pl.requires_grad:  # this rank owns it
                        pg.copy_(pl.data)
            self.local_model.load_state_dict(self.model.state_dict())
            return

        # -------- multi-process merge ----------------------------------------
        with torch.no_grad():
            # 1. Copy owned slices; zero the others
            for pg, pl in zip(self.model.parameters(), self.local_model.parameters()):
                if pl.requires_grad:
                    pg.copy_(pl.data)  # owner rank writes its update
                else:
                    pg.zero_()  # non-owners write zeros

            # 2. Sum across all ranks – only the owner contributes a non-zero value
            for pg in self.model.parameters():
                dist.all_reduce(pg.data, op=dist.ReduceOp.SUM)

        # 3. (optional) broadcast buffers such as BatchNorm running stats
        # for buf in self.model.buffers():
        #     dist.broadcast(buf.data, src=0)

        # 4. Keep the local replica in sync for the next local step
        # self.local_model.load_state_dict(self.model.state_dict())

    def step(self, inputs, labels):
        self.inputs, self.labels = inputs, labels

        with torch.no_grad():
            initial_params = flatten_params(
                self.model, self._flat_params_buffer
            ).clone()

        initial_global_loss = self.global_closure(compute_grad=True)
        initial_local_loss = self.local_closure(compute_grad=True)

        # print("Initial global loss:", initial_global_loss)
        # print("Initial local loss:", initial_local_loss)

        global_grad = parameters_to_vector(
            [p.grad for p in self.model.parameters()]
        ).detach()

        local_grad = trainable_parameters_to_vector(self.local_model)
        local_loss = self.local_optimizer.step(
            closure=self.local_closure, old_loss=initial_local_loss, grad=local_grad
        )
        # Synchronize paramters to build current flat
        self.sync_loc_to_glob()
        current_flat = flatten_params(self.model, self._local_flat_buffer)

        # Step is current parmeters minus initial parameters
        step = current_flat - initial_params

        # Do dogleg or TR control
        if self.dogleg:
            lr = self.lr
            w = 0
            restore_params(self.model, initial_params + step)
            trial_loss = self.global_closure()
            while trial_loss > initial_global_loss and w <= 1:
                with torch.no_grad():
                    lr *= self.global_optimizer.dec_factor
                    w += 0.2
                    step_update = ((1 - w) * step) - (w * global_grad)
                    step_update = (lr / step_update.norm()) * step_update
                    restore_params(self.model, step_update)
                trial_loss = self.global_closure()
                torch.cuda.empty_cache()
        else:  # TR control
            step_norm = step.norm()
            candidate_step = (
                step if step_norm <= self.lr else (self.lr / step_norm) * step
            )

            # Apply the candidate step
            restore_params(self.model, candidate_step)
            trial_loss = self.global_closure()

            actual_reduction = initial_global_loss - trial_loss
            predicted_reduction = torch.dot(global_grad, step) - (0.5 * step_norm**2)
            rho = actual_reduction / predicted_reduction

            if rho < 0.25:
                # Too small step, reduce the step size
                self.lr = max(
                    self.lr * self.global_optimizer.dec_factor,
                    self.global_optimizer.min_lr,
                )
                restore_params(self.model, -candidate_step)
                trial_loss = initial_global_loss
            elif rho > 0.75:
                # Good step, increase the step size
                self.lr = min(
                    self.lr * self.global_optimizer.inc_factor,
                    self.global_optimizer.max_lr,
                )
                restore_params(self.model, candidate_step)

        loss = self.global_optimizer.step(
            closure=self.global_closure, old_loss=trial_loss
        )
        # Local models get global model parameters, as does the local optimizer
        with torch.no_grad():
            self.lr = self.global_optimizer.lr
            self.local_optimizer.lr = self.lr
            self.local_model.load_state_dict(self.model.state_dict())
        return loss.item()
