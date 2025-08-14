import torch
import torch.distributed as dist

from .apts_d import APTS_D


class APTS_PINN(APTS_D):
    __name__ = "APTS_PINN"

    def __init__(
        self,
        *args,
        num_subdomains: int = 1,
        overlap: float = 0.0,
        criterion=None,
        **kwargs,
    ):
        super().__init__(*args, nr_models=num_subdomains, criterion=criterion, **kwargs)
        self.num_subdomains = max(1, int(num_subdomains))
        low = getattr(self.criterion, "low", 0.0)
        high = getattr(self.criterion, "high", 1.0)
        self.domain_low, self.domain_high = float(low), float(high)
        self.subdomain_bounds = torch.linspace(low, high, self.num_subdomains + 1)
        self.overlap = max(float(overlap), 0.0)

    def get_subdomain_bounds(self, idx: int):
        """Return (low, high) bounds for a subdomain with optional overlap."""
        base_low = self.subdomain_bounds[idx].item()
        base_high = self.subdomain_bounds[idx + 1].item()
        if self.overlap > 0.0:
            half = self.overlap / 2.0
            low = max(base_low - half, self.domain_low)
            high = min(base_high + half, self.domain_high)
            return low, high
        return base_low, base_high

    def get_subdomain_mask(self, x: torch.Tensor, idx: int):
        """Build a boolean mask selecting points for subdomain ``idx``."""
        low, high = self.get_subdomain_bounds(idx)
        return (x >= low) & (x <= high)

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        # Initialize the global and local gradient evaluations counters
        self.grad_evals = self.loc_grad_evals = 0.0

        # Build mask for the subdomain
        idx = dist.get_rank() if dist.is_initialized() else 0
        x = inputs.squeeze()
        mask = self.get_subdomain_mask(x, idx)

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

        # print(f"Subdomain {idx}. Inputs {sub_in}")
        n_local = int(sub_in.shape[0]) if sub_in is not None else 0
        N_total = int(full_in.shape[0])
        weight = n_local / max(N_total, 1)

        self.inputs, self.labels = full_in, full_lab
        self.inputs_d, self.labels_d = full_in_d, full_lab_d
        self.hNk = hNk

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()

        # Compute the initial global loss and gradient
        self.criterion.current_x = full_in
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        self.init_glob_grad = self.glob_grad_to_vector()

        # Compute the initial local loss and gradient
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

        step, pred = self.aggregate_loc_steps_and_losses(step, loc_red, weight=weight)

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        self.inputs, self.labels = full_in, full_lab
        self.inputs_d, self.labels_d = full_in_d, full_lab_d
        self.criterion.current_x = full_in
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)

        # ── Sync for next iteration ──
        self.sync_glob_to_loc()
        return loss
