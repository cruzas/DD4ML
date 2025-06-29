import torch

from .base_dataset import BaseDataset

class Poisson2DDataset(BaseDataset):
    """Dataset for 2D Poisson equation ``-\Delta u = f`` on [0,1]^2 with zero boundary conditions."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_interior = 1000
        C.n_boundary_side = 20
        C.low = 0.0
        C.high = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config
        low, high = cfg.low, cfg.high
        # interior points sampled uniformly
        interior = torch.rand(cfg.n_interior, 2) * (high - low) + low
        # boundary points along each side
        t = torch.linspace(low, high, cfg.n_boundary_side)
        left = torch.stack([torch.full_like(t, low), t], dim=1)
        right = torch.stack([torch.full_like(t, high), t], dim=1)
        bottom = torch.stack([t, torch.full_like(t, low)], dim=1)
        top = torch.stack([t, torch.full_like(t, high)], dim=1)
        boundary = torch.cat([left, right, bottom, top], dim=0)
        self.x_interior = interior
        self.x_boundary = boundary
        self.data = torch.cat([interior, boundary], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xy = self.data[idx]
        is_boundary = 1.0 if idx >= len(self.x_interior) else 0.0
        return xy, torch.tensor([is_boundary], dtype=torch.float32)
