import time

import torch
import torch.distributed as dist

from dd4ml.optimizers.trust_region import TrustRegion
from dd4ml.optimizers.utils import get_trust_region_params

class APTS(torch.optim.Optimizer):
    @staticmethod
    def setup_APTS_args(config):    
        # Subdomain optimizer
        config.subdomain_optimizer = torch.optim.SGD
        config.subdomain_optimizer_args = {'lr' : config.learning_rate}
        
        if config.subdomain_optimizer == torch.optim.Adam or config.subdomain_optimizer == torch.optim.AdamW:
            config.subdomain_optimizer_args['betas'] = config.betas
        elif config.subdomain_optimizer == torch.optim.SGD:
            config.subdomain_optimizer_args['momentum'] = 0.9
        
        config.max_subdomain_iters = 3
        
        # Global optimizer
        config.global_optimizer = TrustRegion
        config.global_optimizer_args = get_trust_region_params(config, lr_scale=1.0, max_iter=3)
        
        return config
    
    def __init__(self, model, subdomain_optimizer, subdomain_optimizer_defaults,
                 global_optimizer, global_optimizer_defaults, lr=0.01, max_subdomain_iter=0,
                 dogleg=False, APTS_in_data_sync_strategy='average', step_strategy='mean'):
        super().__init__(model.parameters(), {'lr': lr, 'max_subdomain_iter': max_subdomain_iter, 'dogleg': dogleg})
        # Synchronize non-parameter attributes from the first param group.
        self._sync_attributes_from_param_group()
        self.model = model  # subdomain model
        self.APTS_in_data_sync_strategy = APTS_in_data_sync_strategy.lower()
        self.step_strategy = step_strategy

        if self.step_strategy not in ['weighted_mean', 'mean']:
            raise ValueError('The step strategy must be either "weighted_mean" or "mean".')
        if self.APTS_in_data_sync_strategy not in ['average', 'sum']:
            raise ValueError('The APTS in data synchronization strategy must be either "average" or "sum".')
        if self.APTS_in_data_sync_strategy == 'sum' and dist.get_rank() == 0:
            print('(WARNING) APTS in data "sum" synchronization strategy still has to be tested/verified.')
        if lr <= 0:
            raise ValueError('The learning rate "lr" must be bigger than 0.')

        subdomain_optimizer_defaults.update({'lr': lr})
        self.subdomain_optimizer = subdomain_optimizer(
            params=model.subdomain_params(), **subdomain_optimizer_defaults)
        if 'TrustRegion' in str(global_optimizer):
            self.global_optimizer = global_optimizer(model=model, **global_optimizer_defaults)
        else:
            global_optimizer_defaults.update({'lr': lr})
            self.global_optimizer = global_optimizer(
                params=model.subdomain_params(), **global_optimizer_defaults)
        self.timings = {
            'smoother': 0, 'precond': 0, 'copy_params': 0,
            'step_comp': 0, 'dogleg': 0, 'closure_1': 0, 'closure_2': 0
        }

    def _sync_attributes_from_param_group(self):
        for key, value in self.param_groups[0].items():
            if key != 'params':
                setattr(self, key, value)

    def update_param_group(self):
        for key in self.param_groups[0].keys():
            if key != 'params':
                self.param_groups[0][key] = getattr(self, key)

    def _apply_model_update(self, initial_parameters, step_update):
        for i, p in enumerate(self.model.parameters()):
            p.copy_(initial_parameters.tensor[i] + step_update.tensor[i])

    def get_timings(self):
        timings = self.timings.copy()
        if 'tradam' in str(self.subdomain_optimizer).lower():
            timings.update(self.subdomain_optimizer.get_timings())
        return timings

    def display_avg_timers(self):
        timings = self.get_timings()
        timings.pop('precond', None)
        total_time = sum(timings.values())
        headers = ["Timer", "Time (s)", "Percentage (%)"]
        rows = [
            [key, f'{timings[key]:.4f}', f'{(timings[key]/total_time)*100:.2f}%']
            for key in sorted(timings.keys())
        ]
        col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(3)]
        row_format = '  '.join(f'{{:<{w}}}' for w in col_widths)
        table = [row_format.format(*headers), '-' * sum(col_widths)]
        for row in rows:
            table.append(row_format.format(*row))
        table_str = '\n'.join(table)
        print(table_str)
        return table_str

    def subdomain_steps(self, final_subdomain_closure=None):
        if self.max_subdomain_iter > 0:
            lr_subdomain = self.lr / self.max_subdomain_iter
            self.subdomain_optimizer.param_groups[0]['lr'] = lr_subdomain
            for i in range(self.max_subdomain_iter):
                self.subdomain_optimizer.step()
                self.subdomain_optimizer.zero_grad()
                if i != self.max_subdomain_iter - 1:
                    outputs = self.model.subdomain_forward()
                    losses = (final_subdomain_closure(outputs)
                              if self.model.model_handler.is_last_stage() else [])
                    self.model.subdomain_backward(losses)
            self.model.sync_params(method='average')
            self.update_param_group()

    def zero_timers(self):
        self.timings = {key: 0 for key in self.timings}
        if hasattr(self.subdomain_optimizer, 'zero_timers'):
            self.subdomain_optimizer.zero_timers()

    def step(self, closure, final_subdomain_closure):
        tic = time.time()
        initial_loss = closure(compute_grad=True, zero_grad=True)
        self.timings['closure_1'] += time.time() - tic

        tic = time.time()
        initial_parameters = self.model.parameters(clone=True)
        initial_grads = self.model.grad(clone=True)
        self.timings['copy_params'] += time.time() - tic

        tic = time.time()
        self.subdomain_steps(final_subdomain_closure)
        self.timings['precond'] += time.time() - tic

        with torch.no_grad():
            tic = time.time()
            new_loss = closure(compute_grad=False, zero_grad=True)
            self.timings['closure_2'] += time.time() - tic
            tic = time.time()
            step = self.model.parameters(clone=False) - initial_parameters
            lr = self.lr
            w = 0
            c = 0
            self.timings['step_comp'] += time.time() - tic

        if self.dogleg:
            tic = time.time()
            while new_loss > initial_loss and c < 5:
                with torch.no_grad():
                    c += 1
                    lr /= 2
                    w = min(w + 0.2, 1)
                    step_update = ((1 - w) * step) - (w * initial_grads)
                    step_update = (lr / step_update.norm()) * step_update
                    self._apply_model_update(initial_parameters, step_update)
                new_loss = closure(compute_grad=False, zero_grad=True)
                torch.cuda.empty_cache()
            self.timings['dogleg'] += time.time() - tic
        else:
            with torch.no_grad():
                self._apply_model_update(initial_parameters, step)

        tic = time.time()
        self.global_optimizer.step(closure)
        self.timings['smoother'] += time.time() - tic

        if 'lr' in self.global_optimizer.param_groups[0]:
            self.lr = self.global_optimizer.param_groups[0]['lr']
        else:
            self.lr = self.global_optimizer.lr
        self.update_param_group()
        return new_loss
