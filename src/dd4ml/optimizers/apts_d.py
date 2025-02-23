import copy

import torch
import torch.distributed as dist  # For distributed initialization if needed
from torch.optim.optimizer import Optimizer

from dd4ml.optimizers.trust_region import *


class APTS_D(Optimizer):
    def setup_APTS_args(config):
        # TODO: streamline the specification of the sub/global domain optimizer and their parameters.
        from dd4ml.optimizers.trust_region import TrustRegion
        config.max_subdomain_iters = 3
        # global optimizer
        config.global_optimizer = TrustRegion
        config.global_optimizer_args = {
            'lr': config.learning_rate,
            'max_lr': 10.0,
            'min_lr': 1e-4,
            'nu': 0.5,
            'inc_factor': 2.0,
            'dec_factor': 0.5,
            'nu_1': 0.25,
            'nu_2': 0.75,
            'max_iter': 3, # To decrease gradient directly within the trust region method
            'norm_type': 2
        }
        
        config.subdomain_optimizer = TrustRegion
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        config.subdomain_optimizer_args = {
            'lr': config.learning_rate/world_size,
            'max_lr': 10.0,
            'min_lr': 1e-4,
            'nu': 0.5,
            'inc_factor': 2.0,
            'dec_factor': 0.5,
            'nu_1': 0.25,
            'nu_2': 0.75,
            'max_iter': 3, # To decrease gradient directly within the trust region method
            'norm_type': 2
        }
        
        return config
    
    def __init__(self,
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
                 foc=True):

        super(APTS_D, self).__init__(params, {})

        self.model = model  # This model is assumed to be wrapped in DDP
        if hasattr(model, 'module'):
            self.local_model = copy.deepcopy(model.module)
        else:
            self.local_model = copy.deepcopy(model)
        self.max_iter = max_iter
        self.nr_models = nr_models
        self.device = device if device is not None else (f'cuda:{torch.cuda.current_device()}' if self.backend != 'gloo' else 'cpu')
        self.criterion = criterion
        self.global_pass = bool(global_pass)
        self.foc = bool(foc)

        # Closure inputs
        self.inputs = None
        self.labels = None
        self.residual = None
        self.grad_evals_counter = torch.tensor(0.0, device=self.device)

        if 'TrustRegion' in str(global_opt):
            self.global_optimizer = global_opt(self.model, **global_opt_params)
            self.local_optimizer = local_opt(self.local_model, **local_opt_params)
        else:
            self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
            self.local_optimizer = local_opt(self.local_model.parameters(), **local_opt_params)

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
        global_params = torch.cat([p.view(-1) for p in self.model.parameters()])
        local_params = torch.cat([p.view(-1) for p in self.local_model.parameters()])
        s = local_params - global_params
        if not torch.all(torch.abs(s) < 1e-8):
            local_loss = local_loss + (self.residual @ s)
        return local_loss

    def global_closure(self):
        self.zero_grad()
        outputs = self.model(self.inputs)
        global_loss = self.criterion(outputs, self.labels)
        if torch.is_grad_enabled():
            global_loss.backward()
        return global_loss

    def step(self, inputs, labels):
        self.grad_evals_counter = torch.tensor(0.0, device=self.device)
        self.inputs = inputs
        self.labels = labels
        self.residual = torch.tensor(0.0, device=self.device)
        local_closure = self.foc_local_closure if self.foc else self.non_foc_local_closure
        
        with torch.no_grad():
            initial_params = torch.cat([p.view(-1) for p in self.model.parameters()])
        
        initial_global_loss = self.global_closure()
        self.grad_evals_counter += 1
        initial_local_loss = local_closure()
        
        global_gradient = torch.cat([p.grad.flatten() for p in self.model.parameters()]).detach()
        initial_local_gradient = torch.cat([p.grad.flatten() for p in self.local_model.parameters()]).detach().clone()
        self.residual = global_gradient - initial_local_gradient

        for _ in range(self.max_iter):
            local_loss = self.local_optimizer.step(local_closure)
            self.grad_evals_counter += 1.0 / self.nr_models
            with torch.no_grad():
                step = torch.cat([p.view(-1) for p in self.local_model.parameters()]) - initial_params
                if torch.norm(step, p=2) >= self.local_optimizer.max_lr:
                    break

        with torch.no_grad():
            local_reduction = initial_local_loss - torch.tensor(local_loss, device=self.device)
            a = 0
            for param in self.model.parameters():
                b = param.numel()
                param.data.copy_(torch.reshape(initial_params[a:a+b] + step[a:a+b], param.shape))
                a += b

            trial_loss = self.global_closure()
            acceptance_ratio = (initial_global_loss - trial_loss) / local_reduction

            if acceptance_ratio < self.global_optimizer.reduction_ratio:
                self.lr = max(self.lr * self.global_optimizer.decrease_factor,
                                  self.global_optimizer.min_lr)
                a = 0
                for param in self.model.parameters():
                    b = param.numel()
                    param.data.copy_(torch.reshape(initial_params[a:a+b], param.shape))
                    a += b
                new_loss = initial_global_loss
            elif acceptance_ratio > self.global_optimizer.acceptance_ratio:
                self.lr = min(self.lr * self.global_optimizer.increase_factor,
                                  self.global_optimizer.max_lr)
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
            self.local_model.load_state_dict(self.model.state_dict())

        return new_loss