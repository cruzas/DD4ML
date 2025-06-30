import math
import torch
from .base_dataset import BaseDataset

class SineOperatorDataset(BaseDataset):
    """Synthetic dataset for training DeepONet on functions a -> sin(a * x)."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_samples = 1000
        C.n_sensors = 50
        C.n_trunk = 100
        C.low = 0.0
        C.high = math.pi * 2
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_data()

    def _generate_data(self):
        cfg = self.config
        # Sensor and trunk locations are fixed across samples
        self.sensors = torch.linspace(cfg.low, cfg.high, cfg.n_sensors)
        self.trunk_points = torch.linspace(cfg.low, cfg.high, cfg.n_trunk)
        # Sample random parameters a for each function
        self.a_vals = torch.rand(cfg.n_samples) * (cfg.high - cfg.low) / cfg.high
        self.branch_data = torch.sin(self.a_vals.unsqueeze(1) * self.sensors)

    def __len__(self):
        return self.config.n_samples * self.config.n_trunk

    def __getitem__(self, idx):
        sample_idx = idx // self.config.n_trunk
        trunk_idx = idx % self.config.n_trunk
        a = self.a_vals[sample_idx]
        branch = self.branch_data[sample_idx]
        x = self.trunk_points[trunk_idx]
        y = torch.sin(a * x)
        return (branch, x.unsqueeze(0)), y.unsqueeze(0)
