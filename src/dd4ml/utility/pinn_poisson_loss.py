import torch
import torch.nn as nn

class PoissonPINNLoss(nn.Module):
    """Loss for 1D Poisson PINN (-u'' = f, u(0)=u(1)=0 with f=sin(pi x))."""

    def __init__(self, current_x=None):
        super().__init__()
        self.current_x = current_x

    def forward(self, u_pred, boundary_flag):
        if self.current_x is None:
            raise ValueError("current_x must be set before calling PoissonPINNLoss")
        x = self.current_x
        x.requires_grad_(True)

        grad_u = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
        grad2_u = torch.autograd.grad(grad_u, x, torch.ones_like(grad_u), create_graph=True)[0]

        rhs = torch.sin(torch.pi * x)
        interior = ((-grad2_u.squeeze() - rhs.squeeze()) ** 2) * (1 - boundary_flag.squeeze())
        boundary = (u_pred.squeeze() ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
