import torch
import torch.nn as nn


class AllenCahnTimePINNLoss(nn.Module):
    """Loss for the time-dependent Allen-Cahn PINN.

    This implements the residual for ``u_t - u_xx + u^3 - u = 0`` on
    ``x \in [low_x, high_x]`` and ``t \in [low_t, high_t]`` with boundary
    conditions ``u(low_x, t)=1``, ``u(high_x, t)=-1`` and initial condition
    ``u(x, low_t)=1-2x``.
    """

    def __init__(
        self,
        current_xt: torch.Tensor | None = None,
        low_x: float = 0.0,
        high_x: float = 1.0,
        low_t: float = 0.0,
        high_t: float = 1.0,
        diff_coeff: float = 1.0,
    ):
        super().__init__()
        self.current_xt = current_xt
        self.low_x = low_x
        self.high_x = high_x
        self.low_t = low_t
        self.high_t = high_t
        self.diff_coeff = diff_coeff

    def forward(self, u_pred, boundary_flag):
        if self.current_xt is None:
            raise ValueError("current_xt must be set before calling AllenCahnTimePINNLoss")
        xt = self.current_xt
        if not xt.requires_grad:
            xt.requires_grad_(True)

        # First derivatives
        grad_u = torch.autograd.grad(
            u_pred,
            xt,
            torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Second derivative with respect to x
        grad2_u = torch.autograd.grad(
            u_x,
            xt,
            torch.ones_like(u_x),
            create_graph=True,
        )[0]
        u_xx = grad2_u[:, 0:1]

        residual = u_t - self.diff_coeff * u_xx + u_pred**3 - u_pred
        interior = (residual.squeeze() ** 2) * (1 - boundary_flag.squeeze())

        x = xt[:, 0:1]
        t = xt[:, 1:2]
        bc_left = torch.isclose(
            x.squeeze(), torch.tensor(self.low_x, device=x.device, dtype=x.dtype)
        )
        bc_right = torch.isclose(
            x.squeeze(), torch.tensor(self.high_x, device=x.device, dtype=x.dtype)
        )
        ic = torch.isclose(
            t.squeeze(), torch.tensor(self.low_t, device=t.device, dtype=t.dtype)
        )

        bc_val = torch.zeros_like(u_pred.squeeze())
        bc_val[bc_left] = 1.0
        bc_val[bc_right] = -1.0
        bc_val[ic] = 1 - 2 * x.squeeze()[ic]

        boundary = ((u_pred.squeeze() - bc_val) ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
