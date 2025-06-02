from __future__ import annotations
from typing import Iterable, Tuple
import torch
from torch.optim import Optimizer
from dd4ml.optimizers.lsr1 import LSR1
from dd4ml.solvers.obs import OBS
from dd4ml.utility import get_trust_region_params

class TR(Optimizer):
    @staticmethod
    def setup_TR_args(cfg):
        for k, v in get_trust_region_params(cfg).items():
            setattr(cfg, k, v)
        return cfg

    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        defaults = {**kwargs, 'lr': kwargs.get('delta', 0.1)}
        super().__init__(params, defaults)
        self.ps = [p for g in self.param_groups for p in g['params']]
        # precompute shapes and offsets
        self.shapes = [p.shape for p in self.ps]
        self.numels = [p.numel() for p in self.ps]
        self.offsets = torch.tensor([0] + self.numels).cumsum(0)
        total = int(self.offsets[-1])
        # allocate reusable buffers
        self._grad_buf = torch.zeros(total, device=self.ps[0].device)
        self._step_buf = torch.zeros_like(self._grad_buf)

        group = self.param_groups[0]
        if group['second_order']:
            dev = self.ps[0].device
            self.hess = LSR1(1.0, int(group['mem_length']), dev, group['tol'])
            self.obs = OBS()
        else:
            self.hess = None  # type: ignore
            self.obs = None  # type: ignore

    def _flat_grad(self) -> torch.Tensor:
        self._grad_buf.zero_()
        for i, p in enumerate(self.ps):
            if p.grad is not None:
                start, end = int(self.offsets[i]), int(self.offsets[i + 1])
                self._grad_buf[start:end].copy_(p.grad.view(-1))
        return self._grad_buf

    def _apply_update(self, sign: float = 1.0) -> None:
        with torch.no_grad():
            for i, p in enumerate(self.ps):
                start, end = int(self.offsets[i]), int(self.offsets[i + 1])
                p.add_(self._step_buf[start:end].view(self.shapes[i]) * sign)

    def _solve_tr_first_order(self, g: torch.Tensor, gn: float, delta: float) -> Tuple[torch.Tensor, float]:
        if gn <= self.defaults['tol']:
            return torch.zeros_like(g), 0.0
        step = -g.mul(delta / gn)
        return step, delta * gn

    def _solve_tr_second_order(self, g: torch.Tensor, gn: float, delta: float) -> Tuple[torch.Tensor, float]:
        if gn <= self.defaults["tol"]:
            return torch.zeros_like(g), 0.0
        self.hess.precompute()  # type: ignore
        step = self.obs.solve_tr_subproblem(
            g,
            g.new_tensor(delta),
            self.hess.gamma,
            self.hess.Psi,
            self.hess.Minv,  # type: ignore
        )
        predicted = -(g.dot(step) + 0.5 * step.dot(self.hess.B(step)))  # type: ignore
        return step, predicted.item()

    def step(self, closure, **_) -> Tuple[float, torch.Tensor]:
        group = self.param_groups[0]
        delta = group['delta']
        max_delta = group['max_delta']
        min_delta = group['min_delta']
        inc_factor = group['inc_factor']
        dec_factor = group['dec_factor']
        nu_dec = group['nu_dec']
        nu_inc = group['nu_inc']
        tol = group['tol']
        norm_type = group['norm_type']
        second_order = group['second_order']

        loss_val = _['precomp_loss'] if 'precomp_loss' in _ else closure(compute_grad=True)
        grad = _['precomp_grad'] if 'precomp_grad' in _ else self._flat_grad()
        gn = torch.norm(grad, p=norm_type).item()
        if gn <= tol:
            return loss_val.item(), grad

        use_second = second_order and self.hess and len(self.hess._S) > 0  # type: ignore
        if use_second:
            self._step_buf, predicted = self._solve_tr_second_order(grad, gn, delta)
        else:
            self._step_buf, predicted = self._solve_tr_first_order(grad, gn, delta)

        # apply update
        self._apply_update()
        new_loss = closure(compute_grad=True)
        new_grad = self._flat_grad()

        actual = (loss_val - new_loss).item()
        rho = actual / (predicted + 1e-12)

        if actual > 0 and rho >= nu_dec:
            if second_order:
                self.hess.update_memory(
                    self._step_buf.clone(), (new_grad - grad).clone()  # type: ignore
                )
            if rho >= nu_inc and self._step_buf.norm() >= 0.9 * delta:
                group['delta'] = min(max_delta, inc_factor * delta)
                group['lr'] = group['delta']
            return new_loss, new_grad
        else:
            # revert
            self._apply_update(sign=-1.0)
            group['delta'] = max(min_delta, dec_factor * delta)
            group['lr'] = group['delta']
            return loss_val, grad
