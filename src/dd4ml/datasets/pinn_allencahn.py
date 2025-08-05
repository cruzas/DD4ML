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
