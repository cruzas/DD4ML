import torch
import torch.distributed as dist

from .apts_d import APTS_D


class APTS_PINN(APTS_D):
    __name__ = "APTS_PINN"

    def __init__(self, *args, num_subdomains=1, criterion=None, **kwargs):
        super().__init__(*args, nr_models=num_subdomains, criterion=criterion, **kwargs)
        self.num_subdomains = max(1, int(num_subdomains))
        low = getattr(self.criterion, "low", 0.0)
        high = getattr(self.criterion, "high", 1.0)
        self.subdomain_bounds = torch.linspace(low, high, self.num_subdomains + 1)

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        # Initialize the global and local gradient evaluations counters
        self.grad_evals = self.loc_grad_evals = 0.0

        # Store inputs and labels for ASNTR closures
        self.inputs, self.labels = inputs, labels
        self.inputs_d, self.labels_d = inputs_d, labels_d
        self.hNk = hNk

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()
        # Set the current inputs for the PINN criterion
        self.criterion.current_x = inputs

        # Compute the initial global loss and gradient
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        self.init_glob_grad = self.glob_grad_to_vector()

        # Compute the initial local loss and gradient
        self.init_loc_loss = self.loc_closure(compute_grad=True)
        self.init_loc_grad = self.loc_grad_to_vector()

        # Compute the residual if foc is enabled
        if self.foc:
            self.resid = self.init_glob_grad - self.init_loc_grad

        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)

        # Compute the local step and reduction
        with torch.no_grad():
            step = self.loc_params_to_vector() - self.init_glob_flat
            loc_red = self.init_loc_loss - loc_loss

        step, pred = self.aggregate_loc_steps_and_losses(step, loc_red)

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)

        # ── Sync for next iteration ──
        self.sync_glob_to_loc()
        return loss
