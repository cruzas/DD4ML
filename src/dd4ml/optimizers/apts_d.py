import copy

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer

from dd4ml.optimizers.trust_region import TrustRegion  # Explicit import


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
    # Create a persistent clone of the model.
    if hasattr(model, "module"):
        config_copy = copy.deepcopy(model.module.config)
        config_copy.model_type = None
        new_model = type(model.module)(config_copy)
        new_model.load_state_dict(model.module.state_dict())
    else:
        config_copy = copy.deepcopy(model.config)
        config_copy.model_type = None
        new_model = type(model)(config_copy)
        new_model.load_state_dict(model.state_dict())
    return new_model.to(model.device)


class APTS_D(Optimizer):
    @staticmethod
    def setup_APTS_args(config):
        config.max_subdomain_iters = 3
        # Global optimizer args.
        config.global_optimizer = TrustRegion
        config.global_optimizer_args = {
            "lr": config.learning_rate,
            "max_lr": 10.0,
            "min_lr": 1e-4,
            "nu": 0.5,
            "inc_factor": 2.0,
            "dec_factor": 0.5,
            "nu_1": 0.25,
            "nu_2": 0.75,
            "max_iter": 3,
            "norm_type": 2,
        }
        # Subdomain optimizer args.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        config.subdomain_optimizer = TrustRegion
        config.subdomain_optimizer_args = {
            "lr": config.learning_rate / world_size,
            "max_lr": 10.0,
            "min_lr": 1e-4,
            "nu": 0.5,
            "inc_factor": 2.0,
            "dec_factor": 0.5,
            "nu_1": 0.25,
            "nu_2": 0.75,
            "max_iter": 3,
            "norm_type": 2,
        }
        return config

    def __init__(
        self,
        params,
        model=None,
        criterion=None,
        device=None,
        max_iter=3,
        nr_models=None,
        global_opt=None,
        global_opt_params=None,
        local_opt=None,
        local_opt_params=None,
        global_pass=True,
        foc=True,
    ):
        super().__init__(params, {})
        self.model = model
        # Create a persistent clone once.
        self.local_model = clone_model(model)
        self.max_iter = max_iter
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
        # Preallocate a buffer for flattening parameters.
        sample_flat = parameters_to_vector(model.parameters())
        self._flat_params_buffer = torch.empty_like(sample_flat)
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
        global_flat = flatten_params(self.model, out=self._flat_params_buffer)
        # Use a clone for the local flat to avoid overwriting the buffer.
        local_flat = flatten_params(
            self.local_model, out=self._flat_params_buffer.clone()
        )
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
            loss.backward() # DDP takes care of averaging gradients
        if self.nr_models > 1:
            # Global loss
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
            # Cache the flattened global parameters once.
            initial_flat = flatten_params(
                self.model, out=self._flat_params_buffer
            ).clone()
        initial_global_loss = self.global_closure()
        self.grad_evals_counter += 1
        initial_local_loss = local_closure()

        global_grad = parameters_to_vector(
            [p.grad for p in self.model.parameters()]
        ).detach()
        local_grad = (
            parameters_to_vector([p.grad for p in self.local_model.parameters()])
            .detach()
            .clone()
        )
        self.residual = global_grad - local_grad

        # Compute local corrections 
        step_vec = None
        total_local_grad_evals_counter = torch.tensor(0.0, device=self.device)
        for _ in range(self.max_iter):
            local_loss = self.local_optimizer.step(local_closure)
            total_local_grad_evals_counter += 1
            
            with torch.no_grad():
                # Reuse the preallocated buffer to compute the current flattened parameters.
                current_flat = flatten_params(
                    self.local_model, out=self._flat_params_buffer
                )
                step_vec = current_flat - initial_flat
                if torch.norm(step_vec, p=2) >= self.local_optimizer.max_lr:
                    break
        
        if self.nr_models > 1:
            dist.all_reduce(total_local_grad_evals_counter, op=dist.ReduceOp.SUM)
            total_local_grad_evals_counter /= self.nr_models
        
        # Compute local reduction
        with torch.no_grad():
            local_reduction = initial_local_loss - local_loss.detach().to(self.device)
            
            # Global step is sum of local steps
            if self.nr_models > 1:
                dist.all_reduce(step_vec, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_reduction, op=dist.ReduceOp.SUM)
            
            restore_params(self.model, initial_flat + step_vec)
            trial_loss = self.global_closure()            
            acceptance_ratio = (initial_global_loss - trial_loss) / local_reduction

            if acceptance_ratio < self.global_optimizer.nu_1:
                self.lr = max(
                    self.lr * self.global_optimizer.dec_factor,
                    self.global_optimizer.min_lr,
                )
                restore_params(self.model, initial_flat)
                new_loss = initial_global_loss
            elif acceptance_ratio > self.global_optimizer.nu_2:
                self.lr = min(
                    self.lr * self.global_optimizer.inc_factor,
                    self.global_optimizer.max_lr,
                )
                new_loss = trial_loss
            else:
                new_loss = trial_loss

            self.global_optimizer.lr = self.lr

        if self.global_pass:
            new_loss = self.global_optimizer.step(self.global_closure)
            self.grad_evals_counter += 1

        with torch.no_grad():
            self.lr = self.global_optimizer.lr
            self.local_optimizer.lr = self.lr / self.nr_models
            try:
                # TODO: perhaps just use consume_prefix_in_state_dict_if_present
                state = (
                    self.model.module.state_dict()
                    if hasattr(self.model, "module")
                    else self.model.state_dict()
                )
                self.local_model.load_state_dict(state)
            except RuntimeError:
                print("Local and global models have mismatched state_dict keys.")

        return new_loss
