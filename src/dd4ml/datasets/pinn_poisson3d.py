import torch

from .base_dataset import BaseDataset


class Poisson3DDataset(BaseDataset):
    """Dataset for 3D Poisson equation ``-\Delta u = f`` on [0,1]^3 with zero boundary conditions."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_interior = 1000
        C.n_boundary_side = 10
        C.low = 0.0
        C.high = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config
        low, high = cfg.low, cfg.high
        interior = torch.rand(cfg.n_interior, 3) * (high - low) + low
        t = torch.linspace(low, high, cfg.n_boundary_side)

        yy, zz = torch.meshgrid(t, t, indexing="ij")
        yz = torch.stack([yy.reshape(-1), zz.reshape(-1)], dim=1)
        left = torch.cat([torch.full((yz.size(0), 1), low), yz], dim=1)
        right = torch.cat([torch.full((yz.size(0), 1), high), yz], dim=1)

        xx, zz = torch.meshgrid(t, t, indexing="ij")
        xz = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)
        front = torch.cat([xz[:, 0:1], torch.full((xz.size(0), 1), low), xz[:, 1:2]], dim=1)
        back = torch.cat([xz[:, 0:1], torch.full((xz.size(0), 1), high), xz[:, 1:2]], dim=1)

        xx, yy = torch.meshgrid(t, t, indexing="ij")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        bottom = torch.cat([xy, torch.full((xy.size(0), 1), low)], dim=1)
        top = torch.cat([xy, torch.full((xy.size(0), 1), high)], dim=1)

        boundary = torch.cat([left, right, front, back, bottom, top], dim=0)

        self.x_interior = interior
        self.x_boundary = boundary
        self.data = torch.cat([interior, boundary], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xyz = self.data[idx]
        is_boundary = 1.0 if idx >= len(self.x_interior) else 0.0
        return xyz, torch.tensor([is_boundary], dtype=torch.float32)

