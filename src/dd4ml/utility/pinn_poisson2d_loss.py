import math
import torch
import torch.nn as nn
from .base_pinn_loss import BasePINNLoss

class Poisson2DPINNLoss(BasePINNLoss):
    """Loss for 2D Poisson PINN with forcing ``f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y)`` and zero Dirichlet boundary."""

    def __init__(self, current_xy=None):
        super().__init__(coordinates=current_xy)
        # For backward compatibility
        self.current_xy = current_xy

    @property
    def current_xy(self):
        return self.coordinates

    @current_xy.setter
    def current_xy(self, value):
        self.coordinates = value

    def compute_physics_loss(self, u_pred, coords):
        """Compute the 2D Poisson equation residual: -∇²u - f = 0"""
        # gradients with respect to x and y
        grad_u = self.compute_gradient(u_pred, coords)
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        grad2_u_x = self.compute_second_gradient(u_x, coords)[:, 0:1]
        grad2_u_y = self.compute_second_gradient(u_y, coords)[:, 1:2]
        laplace_u = grad2_u_x + grad2_u_y

        rhs = 2 * math.pi ** 2 * torch.sin(math.pi * coords[:, 0:1]) * torch.sin(math.pi * coords[:, 1:2])
        return (-laplace_u - rhs).squeeze()
