import torch
import torch.distributed as dist

from .apts_d import APTS_D


class APTS_PINN(APTS_D):
    __name__ = "APTS_PINN"

    def __init__(self, *args, num_subdomains=1, criterion=None, **kwargs):
        super().__init__(*args, nr_models=num_subdomains, criterion=criterion, **kwargs)
        self.num_subdomains = int(max(1, num_subdomains))
        low = getattr(self.criterion, "low", 0.0)
        high = getattr(self.criterion, "high", 1.0)
        self.subdomain_bounds = torch.linspace(low, high, self.num_subdomains + 1)

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        if dist.is_initialized():
            subdomain_idx = dist.get_rank() % self.num_subdomains
        else:
            subdomain_idx = 0

        x_vals = inputs.squeeze()
        low = self.subdomain_bounds[subdomain_idx]
        high = self.subdomain_bounds[subdomain_idx + 1]
        mask = (x_vals >= low) & (x_vals <= high)

        sub_inputs = inputs[mask].clone().detach().requires_grad_(True)
        sub_labels = labels[mask] if labels is not None else None
        sub_inputs_d = None
        sub_labels_d = None
        if inputs_d is not None:
            sub_inputs_d = inputs_d[mask].clone().detach().requires_grad_(True)
        if labels_d is not None:
            sub_labels_d = labels_d[mask]
        self.criterion.current_x = sub_inputs
        return super().step(
            sub_inputs, sub_labels, sub_inputs_d, sub_labels_d, hNk
        )

