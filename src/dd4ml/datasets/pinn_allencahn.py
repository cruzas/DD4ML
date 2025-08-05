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

    # ------------------------------------------------------------------
    # Domain decomposition utilities
    # ------------------------------------------------------------------
    def split_domain(self, num_subdomains: int):
        """Split the 1D domain into ``num_subdomains`` contiguous pieces.

        Returns a list of ``AllenCahn1DDataset`` objects, each restricted to
        a sub-interval of ``[low, high]``.  The number of interior points is
        distributed roughly evenly across subdomains while boundary points are
        kept at the global domain boundaries.
        """

        if num_subdomains < 1:
            raise ValueError("num_subdomains must be a positive integer")

        import copy

        cfg = self.config
        subdatasets = []
        step = (cfg.high - cfg.low) / num_subdomains

        for i in range(num_subdomains):
            sub_cfg = copy.deepcopy(cfg)
            sub_cfg.low = cfg.low + i * step
            sub_cfg.high = cfg.low + (i + 1) * step

            # Evenly distribute interior points; keep one boundary point at each
            # end of the global domain. Only the first and last subdomain
            # include the respective global boundary point.
            sub_cfg.n_interior = int(round(cfg.n_interior / num_subdomains))
            sub_cfg.n_boundary = 0
            if i == 0:
                sub_cfg.n_boundary += 1
            if i == num_subdomains - 1:
                sub_cfg.n_boundary += 1

            subdatasets.append(AllenCahn1DDataset(sub_cfg))

        return subdatasets
