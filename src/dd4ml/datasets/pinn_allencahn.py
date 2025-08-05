import math

import torch
import torch.distributed as dist

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
        interior = torch.linspace(
            cfg.low, cfg.high, cfg.n_interior, dtype=torch.float32
        ).unsqueeze(1)
        boundary = torch.tensor([[cfg.low], [cfg.high]], dtype=torch.float32)

        # Determine world size to replicate boundary points across ranks
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        boundary_rep = boundary.repeat(world_size, 1)

        shard_size = math.ceil(len(interior) / world_size)
        data_parts = []
        mask_parts = []
        for r in range(world_size):
            start = r * shard_size
            end = min(start + shard_size, len(interior))
            interior_chunk = interior[start:end]
            b_chunk = boundary_rep[2 * r : 2 * (r + 1)]
            data_parts.append(torch.cat([b_chunk, interior_chunk], dim=0))
            mask_parts.append(
                torch.cat(
                    [torch.ones(len(b_chunk), 1), torch.zeros(len(interior_chunk), 1)],
                    dim=0,
                )
            )

        self.data = torch.cat(data_parts, dim=0)
        self.boundary_mask = torch.cat(mask_parts, dim=0)
        self.x_interior = interior
        self.x_boundary = boundary_rep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        flag = self.boundary_mask[idx]
        return x, flag
