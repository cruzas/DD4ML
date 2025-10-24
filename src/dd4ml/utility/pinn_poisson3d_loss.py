import math
import torch
import torch.nn as nn
from .base_pinn_loss import BasePINNLoss


class Poisson3DPINNLoss(BasePINNLoss):
    """Loss for 3D Poisson PINN with forcing ``f(x,y,z)=3\pi^2\sin(\pi x)\sin(\pi y)\sin(\pi z)`` and zero Dirichlet boundary."""

    def __init__(self, current_xyz=None):
        super().__init__(coordinates=current_xyz)
        # For backward compatibility
        self.current_xyz = current_xyz

    @property
    def current_xyz(self):
        return self.coordinates

    @current_xyz.setter
    def current_xyz(self, value):
        self.coordinates = value

    def compute_physics_loss(self, u_pred, coords):
        """Compute the 3D Poisson equation residual: -∇²u - f = 0"""
        grad_u = self.compute_gradient(u_pred, coords)
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_z = grad_u[:, 2:3]

        grad2_u_x = self.compute_second_gradient(u_x, coords)[:, 0:1]
        grad2_u_y = self.compute_second_gradient(u_y, coords)[:, 1:2]
        grad2_u_z = self.compute_second_gradient(u_z, coords)[:, 2:3]

        laplace_u = grad2_u_x + grad2_u_y + grad2_u_z

        rhs = 3 * math.pi ** 2 * torch.sin(math.pi * coords[:, 0:1]) * torch.sin(math.pi * coords[:, 1:2]) * torch.sin(math.pi * coords[:, 2:3])
        return (-laplace_u - rhs).squeeze()

