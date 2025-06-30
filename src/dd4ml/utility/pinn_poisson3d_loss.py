import math
import torch
import torch.nn as nn


class Poisson3DPINNLoss(nn.Module):
    """Loss for 3D Poisson PINN with forcing ``f(x,y,z)=3\pi^2\sin(\pi x)\sin(\pi y)\sin(\pi z)`` and zero Dirichlet boundary."""

    def __init__(self, current_xyz=None):
        super().__init__()
        self.current_xyz = current_xyz

    def forward(self, u_pred, boundary_flag):
        if self.current_xyz is None:
            raise ValueError("current_xyz must be set before calling Poisson3DPINNLoss")
        xyz = self.current_xyz
        xyz.requires_grad_(True)

        grad_u = torch.autograd.grad(u_pred, xyz, torch.ones_like(u_pred), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_z = grad_u[:, 2:3]

        grad2_u_x = torch.autograd.grad(u_x, xyz, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        grad2_u_y = torch.autograd.grad(u_y, xyz, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        grad2_u_z = torch.autograd.grad(u_z, xyz, torch.ones_like(u_z), create_graph=True)[0][:, 2:3]

        laplace_u = grad2_u_x + grad2_u_y + grad2_u_z

        rhs = 3 * math.pi ** 2 * torch.sin(math.pi * xyz[:, 0:1]) * torch.sin(math.pi * xyz[:, 1:2]) * torch.sin(math.pi * xyz[:, 2:3])
        interior = ((-laplace_u - rhs).squeeze()) ** 2 * (1 - boundary_flag.squeeze())
        boundary = (u_pred.squeeze() ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()

