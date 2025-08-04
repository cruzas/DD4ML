import torch

from .base_dataset import BaseDataset

class AllenCahn1DDataset(BaseDataset):
    """Dataset for 1D Allen-Cahn equation on [0,1] with u(0)=1 and u(1)=-1."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_interior = 100
        C.n_boundary = 2
        C.low = 0.0
        C.high = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config
        interior = torch.linspace(cfg.low, cfg.high, cfg.n_interior, dtype=torch.float32).unsqueeze(1)
        boundary = torch.tensor([[cfg.low], [cfg.high]], dtype=torch.float32)
        self.x_interior = interior
        self.x_boundary = boundary
        self.data = torch.cat([interior, boundary], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        is_boundary = 1.0 if idx >= len(self.x_interior) else 0.0
        return x, torch.tensor([is_boundary], dtype=torch.float32)
