from typing import Callable, Iterable, Union

import torch
from torch.optim import Optimizer


class TRAdam(Optimizer):
    __name__ = "TRAdam"

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        flat_grads_fn: Callable[[], torch.Tensor] | None = None,
        flat_params_fn: Callable[[], torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        # Extract hyperparameters
        self.lr = float(kwargs.pop("lr"))
        self.betas = kwargs.pop("betas", (0.9, 0.999))
        self.eps = float(kwargs.pop("eps", 1e-8))
        self.norm_type = kwargs.pop("norm_type", torch.inf)
        if self.norm_type not in (2, torch.inf):
            raise ValueError("norm_type must be 2 or torch.inf")

        # Hooks for custom flatten operations
        self._flat_grads_fn = flat_grads_fn
        self._flat_params_fn = flat_params_fn

        # Default fields for Optimizer
        defaults = {"lr": self.lr}
        super().__init__(params, defaults)

        # Flattened parameter list and shapes
        self.ps = [p for g in self.param_groups for p in g["params"]]
        self.shapes = [p.shape for p in self.ps]
        self.numels = [p.numel() for p in self.ps]
        self.offsets = torch.tensor([0] + self.numels, device=self.ps[0].device).cumsum(
            0
        )
        total = int(self.offsets[-1])

        # Buffers for gradients, steps, and moments
        device = self.ps[0].device
        self._grad_buf = torch.zeros(total, device=device)
        self._step_buf = torch.zeros_like(self._grad_buf)
        self._m_buf = torch.zeros_like(self._grad_buf)
        self._v_buf = torch.zeros_like(self._grad_buf)
        self.t = 0

    def reset_momentum(self):
        """Resets first and second moment buffers."""
        self._m_buf.zero_()
        self._v_buf.zero_()

    def _flat_grad(self) -> torch.Tensor:
        """Return current gradient as a flat vector."""
        if self._flat_grads_fn is not None:
            g = self._flat_grads_fn()
            return g.clone() if isinstance(g, torch.Tensor) else g
        self._grad_buf.zero_()
        for i, p in enumerate(self.ps):
            if p.grad is not None:
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                self._grad_buf[s:e].copy_(p.grad.view(-1))
        return self._grad_buf.clone()

    def _apply_update(self, sign: float = 1.0) -> None:
        """In-place parameter update: p += sign * step_buf segment."""
        with torch.no_grad():
            if self._flat_params_fn is not None:
                base = self._flat_params_fn()
                updated = base + sign * self._step_buf
                torch.nn.utils.vector_to_parameters(updated, self.ps)
            else:
                for i, p in enumerate(self.ps):
                    s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                    p.add_(self._step_buf[s:e].view(self.shapes[i]) * sign)

    def step(self, closure=None) -> Union[None, float]:
        """Perform a single optimisation step."""
        self.t += 1
        loss = closure() if closure is not None else None
        grad = self._flat_grad()

        # Update biased moments
        self._m_buf.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
        self._v_buf.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

        # Bias-corrected estimates
        m_hat = self._m_buf / (1 - self.betas[0] ** self.t)
        v_hat = self._v_buf / (1 - self.betas[1] ** self.t)

        # Compute step direction
        s = m_hat.div(v_hat.sqrt().add(self.eps))

        # Determine step length
        if self.norm_type == torch.inf:
            step_len = torch.norm(s, p=torch.inf).item()
        else:
            step_len = s.dot(s).item()

        # Scale update
        if step_len > self.lr:
            self._step_buf.copy_(s.mul(self.lr / step_len))
        else:
            self._step_buf.copy_(s.mul(self.lr))

        # Apply update
        self._apply_update(sign=-1.0)
        return loss
