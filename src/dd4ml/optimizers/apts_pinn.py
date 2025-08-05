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
        # Determine which subdomain this rank should process
        if dist.is_initialized():
            idx = dist.get_rank() % self.num_subdomains
        else:
            idx = 0

        # Build mask for the subdomain
        x = inputs.squeeze()
        low, high = self.subdomain_bounds[idx], self.subdomain_bounds[idx + 1]
        mask = (x >= low) & (x <= high)

        # Extract subdomain data for local optimisation
        in_sub = inputs[mask].detach().requires_grad_(True)
        lab_sub = labels[mask] if labels is not None else None
        in_d_sub = (
            inputs_d[mask].detach().requires_grad_(True)
            if inputs_d is not None
            else None
        )
        lab_d_sub = labels_d[mask] if labels_d is not None else None

        # Tell the criterion which x‐values it sees
        self.criterion.current_x = in_sub

        # Run the APTS_D step: local pass sees only subdomain,
        # global pass inside super().step still uses full inputs
        return super().step(
            in_sub, lab_sub, inputs_d=in_d_sub, labels_d=lab_d_sub, hNk=hNk
        )
