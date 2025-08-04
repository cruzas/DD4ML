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
        x = self.current_x
        x.requires_grad_(True)

        grad_u = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
        grad2_u = torch.autograd.grad(grad_u, x, torch.ones_like(grad_u), create_graph=True)[0]

        residual = -grad2_u + u_pred ** 3 - u_pred
        interior = (residual.squeeze() ** 2) * (1 - boundary_flag.squeeze())

        bc_val = torch.where(
            torch.isclose(x.squeeze(), torch.tensor(self.low, device=x.device)),
            torch.tensor(1.0, device=x.device),
            torch.tensor(-1.0, device=x.device),
        )
        boundary = ((u_pred.squeeze() - bc_val) ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
