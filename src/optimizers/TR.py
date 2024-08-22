import torch


class TR(torch.optim.Optimizer):
    def __init__(self, model, criterion, lr=0.01, max_lr=1.0, min_lr=0.0001, nu=0.5, inc_factor=2.0, dec_factor=0.5, nu_1=0.25, nu_2=0.75, max_iter=5, norm_type=2):
        """
        Initializes the TR optimizer.
        Args:
            model (torch.nn.Module): The model to be optimized.
            criterion: The loss function.
            lr (float, optional): The initial learning rate. Defaults to 0.01.
            max_lr (float, optional): The maximum learning rate. Defaults to 1.0.
            min_lr (float, optional): The minimum learning rate. Defaults to 0.0001.
            nu (float, optional): The acceptable reduction ratio. Defaults to 0.5.
            inc_factor (float, optional): The increase factor for the learning rate. Defaults to 2.0.
            dec_factor (float, optional): The decrease factor for the learning rate. Defaults to 0.5.
            nu_1 (float, optional): The lower bound for the reduction ratio. Defaults to 0.25.
            nu_2 (float, optional): The upper bound for the reduction ratio. Defaults to 0.75.
            max_iter (int, optional): The maximum iterations for the TR optimization. Defaults to 5.
            norm_type (int, optional): The type of norm to be used. Defaults to 2.

        We use infinity norm for the gradient norm.
        """
        super().__init__(model.parameters(), {'lr': lr, 'max_lr': max_lr, 'min_lr': min_lr, 'max_iter': max_iter})
        self.model = model
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.criterion = criterion
        self.inc_factor = inc_factor # increase factor for the learning rate
        self.dec_factor = dec_factor # decrease factor for the learning rate
        self.nu_1 = nu_1 # lower bound for the reduction ratio
        self.nu_2 = nu_2 # upper bound for the reduction ratio
        self.nu = min(nu, nu_1) # acceptable reduction ratio (cannot be bigger than nu_1)
        self.max_iter = max_iter # max iterations for the TR optimization
        self.norm_type = norm_type 
    
    def step(self, closure):
        """
        Args:
            closure: A closure function that computes the loss of the model and returns it.
        Returns:
            The loss value before the optimization step.
        """
        # Compute the loss of the model
        old_loss = closure(compute_grad=True)
        # Retrieve the gradient of the model  TODO: check if this is a copy by reference or not (if not we could use param.data -= param.grad ..... below)
        grad = self.model.grad()
        # Compute the norm of the gradient
        grad_norm2 = grad.norm(p=2)
        grad_norm = grad.norm(p=self.norm_type) if self.norm_type != 2 else grad_norm2 
        # Rescale the gradient to e at the edges of the trust region
        if grad_norm <= torch.finfo(torch.float32).eps:
            print(f'Stopping TR due to ||g|| = {grad_norm}.')
            return old_loss 
        with torch.no_grad():
            grad = grad * (self.lr/grad_norm)
        
        # Make sure the loss decreases
        new_loss = torch.inf
        c = 0
        while old_loss - new_loss < 0 and c < self.max_iter:
            stop = True if abs(self.lr - self.min_lr)/self.min_lr < 1e-6 else False # TODO: Adaptive reduction factor -> if large difference in losses maybe divide by 4
            for i,param in enumerate(self.model.parameters()):
                param.data -= grad.tensor[i].data
            new_loss = closure(compute_grad=False)
            old_lr = self.lr
            
            # Compute the ratio of the loss reduction       
            act_red = old_loss - new_loss # actual reduction
            pred_red = self.lr*grad_norm2 # predicted reduction (first order approximation of the loss function)
            red_ratio = act_red / pred_red # reduction ratio
            
            if red_ratio < self.nu_1: # the reduction ratio is too small -> decrease the learning rate
                self.lr = max(self.min_lr, self.dec_factor*self.lr)
            elif red_ratio > self.nu_2: # the reduction ratio is good enough -> increase the learning rate
                self.lr = min(self.max_lr, self.inc_factor*self.lr)
                break
            # Else learning rate remains unchanged
            if stop:
                break
            
            if red_ratio < self.nu:
                # In place of storing initial weights, we go backward from the current position whenever the step gets rejected (This only works with first order approximation of the loss)
                if self.lr != self.min_lr:
                    if c == 0: 
                        grad = grad * (-self.lr/old_lr)
                    else:
                        grad = grad * (self.lr/old_lr)
                else: # self.lr == self.min_lr
                    if c == 0:
                        grad = grad * (-(old_lr-self.lr)/old_lr)
                    else:
                        grad = grad * ((old_lr-self.lr)/old_lr)
            else:
                break
            c += 1
        return new_loss
