from .apts_d import APTS_D
import torch
from torch.nn.utils import vector_to_parameters

class APTS_PINN(APTS_D):
    __name__ = "APTS_PINN"

    def __init__(self, *args, num_subdomains=1, criterion=None, **kwargs):
        super().__init__(*args, criterion=criterion, **kwargs)
        self.num_subdomains = int(max(1, num_subdomains))
        low = getattr(self.criterion, "low", 0.0)
        high = getattr(self.criterion, "high", 1.0)
        self.subdomain_bounds = torch.linspace(low, high, self.num_subdomains + 1)

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        self.inputs = inputs.clone().detach().requires_grad_(True)
        self.labels = labels
        self.inputs_d, self.labels_d = inputs_d, labels_d
        self.hNk = hNk

        # Global initial loss/grad for the entire domain
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0
        self.init_glob_flat = self.glob_params_to_vector()
        self.criterion.current_x = self.inputs
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        init_glob_grad_full = self.glob_grad_to_vector()
        full_inputs = self.inputs
        full_labels = labels

        total_step = torch.zeros_like(self.init_glob_flat)
        total_red = torch.zeros(1, device=self.init_glob_flat.device)

        x_vals = inputs.squeeze()
        for i in range(self.num_subdomains):
            low, high = self.subdomain_bounds[i], self.subdomain_bounds[i + 1]
            mask = (x_vals >= low) & (x_vals <= high)
            if mask.sum() == 0:
                continue
            self.inputs = full_inputs[mask].clone().detach().requires_grad_(True)
            self.labels = labels[mask]
            self.criterion.current_x = self.inputs
            glob_loss = self.glob_closure_main(compute_grad=True)
            glob_grad = self.glob_grad_to_vector()
            init_loc_loss = self.loc_closure(compute_grad=True)
            init_loc_grad = self.loc_grad_to_vector()
            if self.foc:
                self.resid = glob_grad - init_loc_grad
            loc_loss, _ = self.loc_steps(init_loc_loss, init_loc_grad)
            with torch.no_grad():
                step = self.loc_params_to_vector() - self.init_glob_flat
                loc_red = init_loc_loss - loc_loss
            total_step += step
            total_red += loc_red
            vector_to_parameters(self.init_glob_flat, self.model.parameters())
            vector_to_parameters(self.init_glob_flat, self.loc_model.parameters())

        self.inputs, self.labels = full_inputs, full_labels
        self.criterion.current_x = full_inputs
        self.init_glob_grad = init_glob_grad_full

        step, pred = self.aggregate_loc_steps_and_losses(total_step, total_red)
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)
        self.sync_glob_to_loc()
        return loss
