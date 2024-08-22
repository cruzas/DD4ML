import torch
import time

class TRAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, norm_type=torch.inf): #torch.inf
        """
        Initializes a TRAdam optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
            norm_type (float or torch.Tensor, optional): Type of norm to be used. Must be 2 or torch.inf. Defaults to torch.inf.

        Raises:
            ValueError: If norm_type is neither 2 nor torch.inf.
        """

        super(TRAdam, self).__init__(params, {'lr': lr, 'betas': betas, 'eps': eps, 'norm_type': norm_type})
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.norm_type = norm_type
        if norm_type != 2 and norm_type != torch.inf:
            raise ValueError('The norm type must be 2 or torch.inf')
        self.m = [0 for _ in self.param_groups[0]['params']]
        self.v = [0 for _ in self.param_groups[0]['params']]
        self.t = 0
        self.timings = {}
    
    def reset_momentum(self):
        """
        Resets the momentum values for the optimizer.

        This method sets the momentum values (`m` and `v`) to zero for all parameters in the optimizer's parameter groups.
        """
        self.m = [0 for _ in self.param_groups[0]['params']]
        self.v = [0 for _ in self.param_groups[0]['params']]

    def get_timings(self):
        """
        Returns:
            dict: A dictionary containing the timings of the optimizer.
        """
        timings = {key: self.timings[key] for key in self.timings.keys()}
        return timings
    
    def zero_timers(self):
        """
        Resets the timers for each key in the timings dictionary to zero.

        Parameters:
        - self: The instance of the TRAdam optimizer.
        """
        self.timings = {key: 0 for key in self.timings.keys()}
        
    def display_avg_timers(self):
        """
        Display the average timers for each timer in the timings dictionary.
        Returns:
            str: A formatted table displaying the timers, time in seconds, and percentage.
        """
        timings = self.get_timings()
        total_time = sum(timings.values())
        
        # Create headers
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
            
    def step(self ,closure=None):
        def step(self, closure=None):
            """
            Performs a single optimization step.
            Args:
                closure (callable, optional): A closure that reevaluates the model and returns the loss. 
                    If provided, the optimizer will call it before performing the optimization step. 
                    Default: None.
            Returns:
                loss: The loss value after the optimization step.
            Raises:
                RuntimeError: If the gradients are sparse.
            Notes:
                - This method updates the internal parameters of the optimizer.
                - The step length is computed based on the gradients and the optimizer's learning rate.
                - The step length is used to update the model parameters.
            """
        self.t += 1
        tic = time.time()
        loss = closure() if closure is not None else None 
        self.timings['TRAdam_closure_1'] = self.timings.get('TRAdam_closure_1',0) + time.time() - tic
        with torch.no_grad():
            s = [0 for _ in self.param_groups[0]['params']]
            step_length = 0
            for i, p in enumerate(self.param_groups[0]['params']):  # TODO: Run this in parallel through cuda streams ?
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                tic = time.time()
                # Initialize m
                self.m[i] = self.betas[0]*self.m[i] + (1-self.betas[0])*grad
                self.timings['TRAdam_init_m'] = self.timings.get('TRAdam_init_m',0) + time.time() - tic
                # Initialize v
                tic = time.time()
                self.v[i] = self.betas[1]*self.v[i] + (1-self.betas[1])*grad**2
                self.timings['TRAdam_init_v'] = self.timings.get('TRAdam_init_v',0) + time.time() - tic

                tic = time.time()
                m_hat = self.m[i]/(1 - self.betas[0]**self.t) # m_hat
                self.timings['TRAdam_init_mhat'] = self.timings.get('TRAdam_init_mhat',0) + time.time() - tic
                tic = time.time()
                v_hat = self.v[i]/(1 - self.betas[1]**self.t) # v_hat
                self.timings['TRAdam_init_vhat'] = self.timings.get('TRAdam_init_vhat',0) + time.time() - tic
                
                tic = time.time()
                s[i] = m_hat/(v_hat.sqrt() + self.eps)
                self.timings['TRAdam_comp_step'] = self.timings.get('TRAdam_comp_step',0) + time.time() - tic
                
                tic = time.time()
                if self.norm_type == torch.inf:
                    step_length = max(step_length, torch.norm(s[i], p=self.norm_type).item())
                    # step_length = 1 # upper bound for the step length to spare computation time (TODO: Check if this is actually true)
                else:
                    step_length += torch.norm(s[i]).item()**2
                self.timings['TRAdam_step_length'] = self.timings.get('TRAdam_step_length',0) + time.time() - tic
                
            tic = time.time()
            for i,p in enumerate(self.param_groups[0]['params']):
                # p.data -= s[i] * self.lr/step_length if step_length > self.lr and 2==1 else self.lr*s[i]
                p.data -= s[i] * self.lr/step_length if step_length > self.lr else s[i]
            self.timings['TRAdam_take_step'] = self.timings.get('TRAdam_take_step',0) + time.time() - tic
        return loss