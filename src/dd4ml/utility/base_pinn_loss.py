import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePINNLoss(nn.Module, ABC):
    """Base class for PINN loss functions with common functionality."""

    def __init__(self, coordinates=None):
        super().__init__()
        self.coordinates = coordinates

    def validate_coordinates(self, coords_name="coordinates"):
        """Validate that coordinates are set before computation."""
        if self.coordinates is None:
            raise ValueError(f"{coords_name} must be set before calling {self.__class__.__name__}")
        return self.coordinates

    def enable_gradients(self, coords):
        """Enable gradient computation for coordinates."""
        coords.requires_grad_(True)
        return coords

    def compute_gradient(self, output, coords, create_graph=True):
        """Compute first-order gradient of output with respect to coordinates."""
        return torch.autograd.grad(
            output, coords, torch.ones_like(output), create_graph=create_graph
        )[0]

    def compute_second_gradient(self, first_grad, coords, create_graph=True):
        """Compute second-order gradient."""
        return torch.autograd.grad(
            first_grad, coords, torch.ones_like(first_grad), create_graph=create_graph
        )[0]

    def apply_boundary_conditions(self, interior_loss, boundary_loss):
        """Apply boundary conditions by combining interior and boundary losses."""
        return interior_loss.mean() + boundary_loss.mean()

    @abstractmethod
    def compute_physics_loss(self, u_pred, coords):
        """Compute the physics-based loss (PDE residual). Must be implemented by subclasses."""
        pass

    def forward(self, u_pred, boundary_flag):
        """Forward pass with common structure for all PINN losses."""
        coords = self.validate_coordinates()
        coords = self.enable_gradients(coords)

        # Compute physics loss
        physics_residual = self.compute_physics_loss(u_pred, coords)
        interior_loss = (physics_residual ** 2) * (1 - boundary_flag.squeeze())

        # Compute boundary loss
        boundary_loss = (u_pred.squeeze() ** 2) * boundary_flag.squeeze()

        return self.apply_boundary_conditions(interior_loss, boundary_loss)