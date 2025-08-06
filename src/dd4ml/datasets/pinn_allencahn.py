import torch

from .base_dataset import BaseDataset


class AllenCahn1DDataset(BaseDataset):
    """Dataset for 1D Allen-Cahn equation on [0,1] with u(0)=1 and u(1)=-1."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_interior = 10000
        C.n_boundary = 2
        C.low = 0.0
        C.high = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config
        # Interior points
        interior = torch.linspace(
            cfg.low, cfg.high, cfg.n_interior, dtype=torch.float32
        ).unsqueeze(1)
        # Boundary points
        boundary = torch.tensor([[cfg.low], [cfg.high]], dtype=torch.float32)

        # Combine data and masks
        data = torch.cat([boundary, interior], dim=0)
        mask = torch.cat(
            [
                torch.ones(len(boundary), 1),  # boundary flags
                torch.zeros(len(interior), 1),  # interior flags
            ],
            dim=0,
        )

        self.data = data
        self.boundary_mask = mask
        self.x_interior = interior
        self.x_boundary = boundary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        flag = self.boundary_mask[idx]
        return x, flag
