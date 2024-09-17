import torch
import time
import torch.distributed as dist


class APTS(torch.optim.Optimizer):
    def __init__(self, model, subdomain_optimizer, subdomain_optimizer_defaults, global_optimizer, global_optimizer_defaults, lr=0.01, max_subdomain_iter=0, dogleg=False, APTS_in_data_sync_strategy='average', step_strategy='mean'):
        super(APTS, self).__init__(model.parameters(), {'lr': lr, 'max_subdomain_iter': max_subdomain_iter, 'dogleg': dogleg})
        for key in self.param_groups[0].keys():  
            if key not in ['params']:
                setattr(self, key, self.param_groups[0][key])
        self.model = model # subdomain model
        self.APTS_in_data_sync_strategy = APTS_in_data_sync_strategy.lower()
        self.step_strategy = step_strategy
            
        if self.step_strategy not in ['weighted_mean', 'mean']:
            raise ValueError('The step strategy must be either "weighted_mean" or "mean".')
        if self.APTS_in_data_sync_strategy not in ['average', 'sum']:
            raise ValueError('The APTS in data synchronization strategy must be either "average" or "sum"')
        if self.APTS_in_data_sync_strategy == 'sum' and dist.get_rank() == 0:
            print('(WARNING) APTS in data "sum" synchronization strategy still has to be tested/verified.')
        if lr <= 0:
            raise ValueError('The learning rate "lr" must be bigger than 0.')
        subdomain_optimizer_defaults.update({'lr': lr})
        self.subdomain_optimizer = subdomain_optimizer(params=model.subdomain_params(), **subdomain_optimizer_defaults) # subdomain optimizer
        if 'TR' in str(global_optimizer):
            self.global_optimizer = global_optimizer(model=model, **global_optimizer_defaults) # TR optimizer
        else:
            global_optimizer_defaults.update({'lr': lr})
            self.global_optimizer = global_optimizer(params=model.subdomain_params(), **global_optimizer_defaults) # standard PyTorch optimizers
        self.timings = {'smoother':0, 'precond':0, 'copy_params': 0, 'step_comp':0, 'dogleg':0, 'closure_1':0, 'closure_2':0}
   
    def get_timings(self):
        timings = {key: self.timings[key] for key in self.timings.keys()}
        if 'tradam' in str(self.subdomain_optimizer).lower():
            timings.update(self.subdomain_optimizer.get_timings())
        return timings

    def display_avg_timers(self):
        timings = self.get_timings()
        timings.pop('precond')
        total_time = sum(timings.values())
        headers = ["Timer", "Time (s)", "Percentage (%)"]
        
        # Create rows for each timer
        rows = [
            [key, f'{timings[key]:.4f}', f'{(timings[key]/total_time)*100:.2f}%']
            for key in sorted(timings.keys())
        ]
        
        # Find the maximum width of each column
        col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(3)]
        
        # Create a format string for each row
        row_format = '  '.join(f'{{:<{col_width}}}' for col_width in col_widths)
        
        # Create the table
        table = []
        table.append(row_format.format(*headers))
        table.append('-' * sum(col_widths))
        for row in rows:
            table.append(row_format.format(*row))
        
        print('\n'.join(table))
        return '\n'.join(table)
        
    def update_param_group(self):
        for key in self.param_groups[0].keys():
            if key not in ['params']:
                self.param_groups[0][key] = getattr(self, key)
            
    def subdomain_steps(self, closure=None):
        """
        Perform subdomain steps.

        This method sets up the learning rate based on the maximum subdomain iterations and the APTS in-data synchronization strategy.
        It then iterates over the maximum subdomain iterations and performs the subdomain optimization steps.
        After each step, the subdomain optimizer's gradients are zeroed.
        If the TRAdam optimizer is used, the momentum is reset.
        If the model is not using data parallelism, the parameters are synchronized based on the APTS in-data synchronization strategy.
        Finally, the parameter group is updated.

        Parameters:
        - self: The APTS optimizer instance.
        """
        # Set up the learning rate
        if self.max_subdomain_iter > 0:
            if self.APTS_in_data_sync_strategy == 'sum' and self.step_strategy != 'weighted_mean':
                self.subdomain_optimizer.param_groups[0]['lr'] = self.lr/(self.model.num_subdomains*self.max_subdomain_iter)
            else:
                self.subdomain_optimizer.param_groups[0]['lr'] = self.lr/self.max_subdomain_iter
            # Do subdomain steps
            for i in range(self.max_subdomain_iter):
                # TODO: Add criterion to exit for-loop
                self.subdomain_optimizer.step()
                self.subdomain_optimizer.zero_grad()
                if i != self.max_subdomain_iter - 1:
                    self.model.subdomain_forward() 
                    self.model.subdomain_backward()
                    # self.model.normalize_grads() # through infinity norm
                
            # Check if TRAdam is used as the local optimizer
            if 'tradam' in str(self.subdomain_optimizer).lower():
                self.subdomain_optimizer.reset_momentum()
                
            if self.step_strategy == 'weighted_mean' and self.model.num_subdomains > 1:
                # Loss seems to be synchronized because of data_and_weight_parallelized_subdomain
                loss = closure(compute_grad=False, zero_grad=False, sync_loss='local')
                loss = torch.tensor(loss).cuda()
                ds_loss = loss.clone()
                if self.model.rank in self.model.subdomain_final_stages_main_rank:
                    dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=self.model.subdomain_final_stages_main_rank_group) # sums losses across all subdomains
                dist.broadcast(tensor=loss, src=self.model.last_layer_main_shard, group=self.model.subdomain_ranks_group)
                ds_loss_weight = ds_loss/loss
                # Here premultiply every param by ds_loss weight
                for param in self.model.parameters():
                    param.data *= ds_loss_weight
                self.model.sync_params(method='sum')
            else:
                self.model.sync_params(method='average') # if this is "sum" it doesn't work `\_(ツ)_/`

            self.update_param_group()

    def zero_timers(self):
        """
        Resets the timers for the optimizer.

        This method sets all the timers in the `timings` dictionary to zero. If the `subdomain_optimizer` object has a `zero_timers` method, it will also be called.
        """
        self.timings = {key: 0 for key in self.timings.keys()}
        if 'zero_timers' in dir(self.subdomain_optimizer):
            self.subdomain_optimizer.zero_timers()

    def step(self, closure):
        """
        Performs a single optimization step.
        Args:
            closure (callable): A closure that re-evaluates the model and returns the loss.
        Returns:
            float: The new loss after the optimization step.
        """
        # TODO: Seed for dropout layers
        # Compute loss
        tic = time.time()
        initial_loss = closure(compute_grad=True, zero_grad=True)
        self.timings['closure_1'] += time.time() - tic
        
        # Store the initial parameters and gradients
        tic = time.time()
        initial_parameters = self.model.parameters(clone=True)
        initial_grads = self.model.grad(clone=True)
        self.timings['copy_params'] += time.time() - tic
        
        # Do subdomain steps
        tic = time.time()
        self.subdomain_steps(closure)
            
        self.timings['precond'] += time.time() - tic
        with torch.no_grad():
            tic = time.time()
        new_loss = closure(compute_grad=False, zero_grad=True)
        with torch.no_grad():
            self.timings['closure_2'] += time.time() - tic
            tic = time.time()
            step = self.model.parameters(clone=False) - initial_parameters
            # Compute the dogleg step with the hope that new_loss <= old_loss
            lr = self.lr
            w = 0; c = 0
            self.timings['step_comp'] += time.time() - tic
            tic = time.time()
        if self.dogleg:
            while new_loss > initial_loss and c<5: 
                with torch.no_grad():
                    c += 1
                    # Decrease lr to decrease size of step ...
                    lr = lr/2
                    # ... while moving towards the steepest descent direction (-g)
                    w = min(w + 0.2, 1)
                    step2 = ((1-w)*step) - (w*initial_grads)
                    # The step length is "lr", with   lr <= self.lr (global TR lr)
                    step2 = (lr/step2.norm())*step2
                    # Update the model with the new params
                    for i,p in enumerate(self.model.parameters()):
                        p.copy_(initial_parameters.tensor[i] + step2.tensor[i])
                    # Compute new global loss
                new_loss = closure(compute_grad=False, zero_grad=True)
                # Empty cache to avoid memory problems
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                # Update the model with the new params
                for i,p in enumerate(self.model.parameters()):
                    p.copy_(initial_parameters.tensor[i] + step.tensor[i])
                self.timings['dogleg'] += time.time() - tic

        # Do global TR step
        tic = time.time()
        self.global_optimizer.step(closure)   
        self.timings['smoother'] += time.time() - tic
         
        # Update the learning rate
        self.lr = self.global_optimizer.lr
        # self.lr = self.global_optimizer.param_groups[0]['lr']
        self.update_param_group()
        return new_loss
