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

        # Combine data and masks and sort by the coordinate value so that
        # downstream consumers (e.g. different ranks in a distributed run)
        # see points in increasing order.
        data = torch.cat([boundary, interior], dim=0)
        mask = torch.cat(
            [
                torch.ones(len(boundary), 1),  # boundary flags
                torch.zeros(len(interior), 1),  # interior flags
            ],
            dim=0,
        )
        sort_idx = torch.argsort(data[:, 0])
        self.data = data[sort_idx]
        self.boundary_mask = mask[sort_idx]
        self.x_interior = interior
        self.x_boundary = boundary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        flag = self.boundary_mask[idx]
        return x, flag

    def split_domain(self, num_subdomains: int, exclusive: bool = False):
        """Split the dataset into ``num_subdomains`` smaller datasets.

        Unlike the previous implementation which re-sampled interior points for
        every subdomain, this version partitions the *existing* dataset so that
        the union of subdomains contains exactly the same points as the original
        dataset. This is useful when different optimizers are expected to operate
        on identical training points.

        The domain ``[low, high]`` is divided into equal sub-intervals. Each
        subdomain receives its own copy of the configuration with updated
        ``low``/``high`` bounds but inherits the data points that fall within the
        interval. If ``exclusive`` is ``True`` the boundary at the beginning of a
        subdomain (except the first) is excluded so that adjacent subdomains do
        not share points.

        Args:
            num_subdomains: Number of contiguous subdomains to create.
            exclusive: If ``True``, adjacent subdomains will not share boundary
                points.
        """

        if num_subdomains < 1:
            raise ValueError("num_subdomains must be at least 1")

        cfg = self.config
        interval = (cfg.high - cfg.low) / num_subdomains

        subdomains = []
        for i in range(num_subdomains):
            sub_low = cfg.low + i * interval
            sub_high = sub_low + interval

            if exclusive and i < num_subdomains - 1:
                mask = (self.data[:, 0] >= sub_low) & (self.data[:, 0] < sub_high)
            else:
                mask = (self.data[:, 0] >= sub_low) & (self.data[:, 0] <= sub_high)

            sub_data = self.data[mask]
            sub_mask = self.boundary_mask[mask]

            sub_cfg = CfgNode(
                n_interior=int((sub_mask == 0).sum().item()),
                n_boundary=int((sub_mask == 1).sum().item()),
                low=sub_low,
                high=sub_high,
            )

            # Bypass __init__ to avoid regenerating points.
            sub_ds = AllenCahn1DDataset.__new__(AllenCahn1DDataset)
            BaseDataset.__init__(sub_ds, sub_cfg)
            sub_ds.data = sub_data
            sub_ds.boundary_mask = sub_mask
            sub_ds.x_interior = sub_data[sub_mask.squeeze() == 0]
            sub_ds.x_boundary = sub_data[sub_mask.squeeze() == 1]
            subdomains.append(sub_ds)

        return subdomains
