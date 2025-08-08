import torch
import torch.nn as nn


class AllenCahnPINNLoss(nn.Module):
    """Loss for 1D Allen-Cahn PINN (-u'' + u^3 - u = 0, u(0)=1, u(1)=-1)."""

    def __init__(self, current_x=None, low=0.0, high=1.0):
        super().__init__()
        self.current_x = current_x
        self.low = low
        self.high = high

    def forward(self, u_pred, boundary_flag):
        if self.current_x is None:
            raise ValueError("current_x must be set before calling AllenCahnPINNLoss")

        # OPTION 1 (RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.)
        x = self.current_x
        # x.requires_grad_(True)

        # OPTION 2 (leads to RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.)
        # x = self.current_x.detach().requires_grad_(True)

        grad_u = torch.autograd.grad(
            u_pred, x, torch.ones_like(u_pred), create_graph=True
        )[0]
        grad2_u = torch.autograd.grad(
            grad_u, x, torch.ones_like(grad_u), create_graph=True
        )[0]

        residual = -grad2_u + u_pred**3 - u_pred
        interior = (residual.squeeze() ** 2) * (1 - boundary_flag.squeeze())

        # Only compute bc_val for actual boundary points
        bc_left = torch.isclose(x.squeeze(), torch.tensor(self.low, device=x.device))
        bc_right = torch.isclose(x.squeeze(), torch.tensor(self.high, device=x.device))

        bc_val = torch.zeros_like(x.squeeze())
        bc_val[bc_left] = 1.0
        bc_val[bc_right] = -1.0

        # The boundary_flag mask will ensure only actual boundary points contribute
        boundary = ((u_pred.squeeze() - bc_val) ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
