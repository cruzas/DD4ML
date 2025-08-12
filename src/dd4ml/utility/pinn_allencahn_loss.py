import torch
import torch.nn as nn


class AllenCahnPINNLoss(nn.Module):
    """Loss for 1D Allen-Cahn PINN (-u'' + u^3 - u = 0, u(0)=1, u(1)=-1)."""

    def __init__(self, current_x=None, low=0.0, high=1.0):
        super().__init__()
        self.current_x = current_x
        self.low = low
        self.high = high

    def forward(self, u_pred, boundary_flag):
        if self.current_x is None:
            raise ValueError("current_x must be set before calling AllenCahnPINNLoss")
        x = self.current_x
        # ``current_x`` is expected to be the tensor that was used to produce
        # ``u_pred``. It should already require gradients (the optimizer sets
        # ``requires_grad`` and detaches it from any previous graph).  However,
        # in case it was provided without gradient tracking, enable it so that
        # autograd can compute derivatives of ``u_pred`` with respect to ``x``.
        if not x.requires_grad:
            x.requires_grad_(True)

        # Compute first and second derivatives of the prediction w.r.t. ``x``.
        # ``retain_graph=True`` is required for the first derivative so that the
        # computation graph remains available when taking the second derivative.
        grad_u = torch.autograd.grad(
            u_pred,
            x,
            torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad2_u = torch.autograd.grad(
            grad_u,
            x,
            torch.ones_like(grad_u),
            create_graph=True,
        )[0]

        residual = -grad2_u + u_pred**3 - u_pred
        interior = (residual.squeeze() ** 2) * (1 - boundary_flag.squeeze())

        # Only compute bc_val for actual boundary points.  ``torch.tensor`` defaults
        # to ``float32`` which causes dtype mismatches when ``x`` is ``float64``.
        # Explicitly match the dtype of ``x`` so ``torch.isclose`` works for both
        # single- and double-precision tensors.
        bc_left = torch.isclose(
            x.squeeze(),
            torch.tensor(self.low, device=x.device, dtype=x.dtype),
        )
        bc_right = torch.isclose(
            x.squeeze(),
            torch.tensor(self.high, device=x.device, dtype=x.dtype),
        )

        bc_val = torch.zeros_like(x.squeeze())
        bc_val[bc_left] = 1.0
        bc_val[bc_right] = -1.0

        # The boundary_flag mask will ensure only actual boundary points contribute
        boundary = ((u_pred.squeeze() - bc_val) ** 2) * boundary_flag.squeeze()
        return interior.mean() + boundary.mean()
