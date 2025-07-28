import math
import torch
import torch.nn as nn

class Poisson2DPINNLoss(nn.Module):
    """Loss for 2D Poisson PINN with forcing ``f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y)`` and zero Dirichlet boundary."""

    def __init__(self, current_xy=None):
        super().__init__()
        self.current_xy = current_xy

    def forward(self, u_pred, boundary_flag):
        if self.current_xy is None:
            raise ValueError("current_xy must be set before calling Poisson2DPINNLoss")
        xy = self.current_xy
        xy.requires_grad_(True)

        # gradients with respect to x and y
        grad_u = torch.autograd.grad(u_pred, xy, torch.ones_like(u_pred), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        grad2_u_x = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        grad2_u_y = torch.autograd.grad(u_y, xy, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        laplace_u = grad2_u_x + grad2_u_y

        rhs = 2 * math.pi ** 2 * torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])
        interior = ((-laplace_u - rhs).squeeze()) ** 2 * (1 - boundary_flag.squeeze())
        boundary = (u_pred.squeeze() ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
