import torch
import torch.nn as nn
from .base_pinn_loss import BasePINNLoss

class PoissonPINNLoss(BasePINNLoss):
    """Loss for 1D Poisson PINN (-u'' = f, u(0)=u(1)=0 with f=sin(pi x))."""

    def __init__(self, current_x=None):
        super().__init__(coordinates=current_x)
        # For backward compatibility
        self.current_x = current_x

    @property
    def current_x(self):
        return self.coordinates

    @current_x.setter
    def current_x(self, value):
        self.coordinates = value

    def compute_physics_loss(self, u_pred, coords):
        """Compute the 1D Poisson equation residual: -u'' - f = 0"""
        grad_u = self.compute_gradient(u_pred, coords)
        grad2_u = self.compute_second_gradient(grad_u, coords)

        rhs = torch.sin(torch.pi * coords)
        return (-grad2_u.squeeze() - rhs.squeeze())
