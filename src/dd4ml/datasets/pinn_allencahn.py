import torch

from dd4ml.utility import CfgNode

from .base_dataset import BaseDataset


class AllenCahn1DDataset(BaseDataset):
    """Dataset for 1D Allen-Cahn equation on [0,1] with u(0)=1 and u(1)=-1."""

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.n_interior = 10
        C.n_boundary = 2
        C.low = 0.0
        C.high = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config
        # Interior points exclude boundaries
        step = (cfg.high - cfg.low) / (cfg.n_interior + 1)
        interior = torch.linspace(
            cfg.low + step, cfg.high - step, cfg.n_interior, dtype=torch.float32
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

    def split_domain(self, num_subdomains: int):
        """Split the dataset into ``num_subdomains`` smaller datasets.

        The domain ``[low, high]`` is divided into equal sub-intervals.  Each
        subdomain receives its own copy of the configuration with updated
        ``low``/``high`` bounds and an evenly distributed number of interior
        points.  Boundary points are placed at the ends of each subdomain.
        """

        if num_subdomains < 1:
            raise ValueError("num_subdomains must be at least 1")

        cfg = self.config
        total_interior = cfg.n_interior
        base_interior = total_interior // num_subdomains
        remainder = total_interior % num_subdomains
        interval = (cfg.high - cfg.low) / num_subdomains

        subdomains = []
        start = cfg.low
        for i in range(num_subdomains):
            # Distribute any remainder one by one to the first subdomains
            n_int = base_interior + (1 if i < remainder else 0)

            sub_cfg = CfgNode(
                n_interior=n_int,
                n_boundary=2,
                low=start,
                high=start + interval,
            )

            subdomains.append(AllenCahn1DDataset(sub_cfg))
            start += interval

        return subdomains
