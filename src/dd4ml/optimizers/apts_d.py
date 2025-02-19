import copy

import torch
import torch.distributed as dist  # For distributed initialization if needed
from torch.optim.optimizer import Optimizer

from optimizers.trust_region import *


class APTS_D(Optimizer):
    def __init__(self,
                 params,
                 model=None,
                 loss_fn=None,
                 device=None,
                 max_iter=5, 
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
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.loss_fn = loss_fn
        self.global_pass = bool(global_pass)
        self.foc = bool(foc)

        # Closure inputs
        self.inputs = None
        self.labels = None
        self.residual = None
        self.grad_evals_counter = torch.tensor(0.0, device=self.device)

        self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
        self.local_optimizer = local_opt(self.local_model.parameters(), **local_opt_params)

        self.radius = self.global_optimizer.radius

    def non_foc_local_closure(self):
        self.local_optimizer.zero_grad()
        outputs = self.local_model(self.inputs)
        loss = self.loss_fn(outputs, self.labels)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    def foc_local_closure(self):
        local_loss = self.non_foc_local_closure()
        global_params = torch.cat([p.view(-1) for p in self.model.parameters()])
        local_params = torch.cat([p.view(-1) for p in self.local_model.parameters()])
        s = local_params - global_params
        if not torch.all(torch.abs(s) < 1e-8):
            local_loss = local_loss + (self.residual @ s)
        return local_loss

    def global_closure(self):
        self.zero_grad()
        outputs = self.model(self.inputs)
        global_loss = self.loss_fn(outputs, self.labels)
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
            local_loss, _, _ = self.local_optimizer.step(local_closure)
            self.grad_evals_counter += 1.0 / self.nr_models
            with torch.no_grad():
                step = torch.cat([p.view(-1) for p in self.local_model.parameters()]) - initial_params
                if torch.norm(step, p=2) >= self.local_optimizer.max_radius:
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
                self.radius = max(self.radius * self.global_optimizer.decrease_factor,
                                  self.global_optimizer.min_radius)
                a = 0
                for param in self.model.parameters():
                    b = param.numel()
                    param.data.copy_(torch.reshape(initial_params[a:a+b], param.shape))
                    a += b
                new_loss = initial_global_loss
            elif acceptance_ratio > self.global_optimizer.acceptance_ratio:
                self.radius = min(self.radius * self.global_optimizer.increase_factor,
                                  self.global_optimizer.max_radius)
                new_loss = trial_loss
            else:
                new_loss = trial_loss

            self.global_optimizer.radius = self.radius

        if self.global_pass:
            new_loss, _, _ = self.global_optimizer.step(self.global_closure)
            self.grad_evals_counter += 1

        with torch.no_grad():
            self.radius = self.global_optimizer.radius
            self.local_optimizer.radius = self.radius / self.nr_models
            self.local_model.load_state_dict(self.model.state_dict())

        return new_loss