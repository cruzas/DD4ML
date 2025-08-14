import torch

from dd4ml.utility import CfgNode

from .base_dataset import BaseDataset


class AllenCahn1DTimeDataset(BaseDataset):
    """Dataset for the time-dependent 1D Allen-Cahn equation.

    Generates collocation points for the spatial domain ``[low_x, high_x]`` and
    time domain ``[low_t, high_t]``. Points are sampled on a regular grid for
    interior, spatial boundary, and initial conditions. Each sample is returned
    as ``(x, t, flag)`` where ``flag`` is ``1`` for boundary/initial points and
    ``0`` for interior points.
    """

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.nx_interior = 32
        C.nt_interior = 32
        C.n_boundary_t = 32
        C.n_initial_x = 32
        C.low_x = 0.0
        C.high_x = 1.0
        C.low_t = 0.0
        C.high_t = 1.0
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        self._generate_points()

    def _generate_points(self):
        cfg = self.config

        # Interior grid excluding boundaries
        if cfg.nx_interior > 0 and cfg.nt_interior > 0:
            x_int = torch.linspace(cfg.low_x, cfg.high_x, cfg.nx_interior + 2)[1:-1]
            t_int = torch.linspace(cfg.low_t, cfg.high_t, cfg.nt_interior + 2)[1:-1]
            xx, tt = torch.meshgrid(x_int, t_int, indexing="ij")
            interior = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
        else:
            interior = torch.empty((0, 2), dtype=torch.float32)

        # Spatial boundaries for all time samples
        t_b = torch.linspace(cfg.low_t, cfg.high_t, cfg.n_boundary_t)
        xb_left = torch.full((cfg.n_boundary_t,), cfg.low_x)
        xb_right = torch.full((cfg.n_boundary_t,), cfg.high_x)
        boundary_left = torch.stack([xb_left, t_b], dim=1)
        boundary_right = torch.stack([xb_right, t_b], dim=1)
        boundary = torch.cat([boundary_left, boundary_right], dim=0)

        # Initial condition (t = low_t) across spatial domain
        x_i = torch.linspace(cfg.low_x, cfg.high_x, cfg.n_initial_x)
        t_i = torch.full((cfg.n_initial_x,), cfg.low_t)
        initial = torch.stack([x_i, t_i], dim=1)

        data = torch.cat([interior, boundary, initial], dim=0)
        mask = torch.cat(
            [
                torch.zeros(len(interior), 1),  # interior flag
                torch.ones(len(boundary), 1),   # spatial boundary flag
                torch.ones(len(initial), 1),    # initial condition flag
            ],
            dim=0,
        )

        self.data = data
        self.x = data[:, 0:1]
        self.t = data[:, 1:2]
        self.boundary_mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.x[idx]
        t = self.t[idx]
        flag = self.boundary_mask[idx]
        return x, t, flag
