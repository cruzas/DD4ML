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

        # Build mask for the subdomain
        idx = dist.get_rank() if dist.is_initialized() else 0
        x = inputs.squeeze()
        low, high = self.subdomain_bounds[idx], self.subdomain_bounds[idx + 1]
        mask = (x >= low) & (x <= high)

        # Create tensors for full domain (for global pass) and subdomain (for local pass)
        full_in = inputs.detach().requires_grad_(True)
        full_lab = labels
        full_in_d = (
            inputs_d.detach().requires_grad_(True) if inputs_d is not None else None
        )
        full_lab_d = labels_d

        sub_in = full_in[mask]
        sub_lab = full_lab[mask] if full_lab is not None else None
        sub_in_d = full_in_d[mask] if full_in_d is not None else None
        sub_lab_d = full_lab_d[mask] if full_lab_d is not None else None

        # ------------------------------------------------------------------
        # Global: evaluate loss/grad on full domain
        # ------------------------------------------------------------------
        self.inputs, self.labels = full_in, full_lab
        self.inputs_d, self.labels_d = full_in_d, full_lab_d
        self.hNk = hNk

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()

        # Set the current inputs for the PINN criterion
        self.criterion.current_x = full_in

        # Compute the initial global loss and gradient
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        self.init_glob_grad = self.glob_grad_to_vector()

        # ------------------------------------------------------------------
        # Local: evaluate loss/grad on subdomain
        # ------------------------------------------------------------------
        self.inputs, self.labels = sub_in, sub_lab
        self.inputs_d, self.labels_d = sub_in_d, sub_lab_d
        self.criterion.current_x = sub_in

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
