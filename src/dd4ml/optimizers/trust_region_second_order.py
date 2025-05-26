import math

import torch
from torch.optim import Optimizer

from dd4ml.utility import get_trust_region_params


class TrustRegionSecondOrder(Optimizer):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

    @staticmethod
    def setup_TR_args(config):
        params = get_trust_region_params(config)
        config.max_iter = params["max_iter"]
        config.lr = params["lr"]
        config.max_lr = params["max_lr"]
        config.min_lr = params["min_lr"]
        config.nu = params["nu"]
        config.inc_factor = params["inc_factor"]
        config.dec_factor = params["dec_factor"]
        config.nu_1 = params["nu_1"]
        config.nu_2 = params["nu_2"]
        config.norm_type = params["norm_type"]
        return config

    def __init__(
        self,
        model,
        lr=0.01,
        max_lr=1.0,
        min_lr=0.0001,
        nu=0.5,
        inc_factor=2.0,
        dec_factor=0.5,
        nu_1=0.25,
        nu_2=0.75,
        max_iter=5,
        norm_type=2,
    ):
        super().__init__(
            model.parameters(),
            {"lr": lr, "max_lr": max_lr, "min_lr": min_lr, "max_iter": max_iter},
        )
        self.model = model
        self.param_list = list(model.parameters())
        self.lr = lr  # Current trust-region radius (step size scaling)
        self.max_lr = max_lr  # Maximum allowed trust-region radius
        self.min_lr = min_lr  # Minimum allowed trust-region radius
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.nu_1 = nu_1  # Acceptance threshold (lower)
        self.nu_2 = nu_2  # Acceptance threshold (upper for expansion)
        self.nu = min(nu, nu_1)
        self.max_iter = max_iter  # Maximum memory size for curvature pairs
        self.norm_type = norm_type
        self.model_has_grad = hasattr(self.model, "grad")
        # Additional state for LSR1 Hessian approximation:
        self.init_hessian_diag = 1.0  # B0 = I scaled by this constant
        self.state["s_history"] = []  # List to store past parameter differences (s)
        self.state["u_history"] = (
            []
        )  # List to store past curvature differences (u = y - B0 * s)
        self.state["prev_grad"] = None
        self.state["prev_loss"] = None
        self.state["prev_params"] = None
        self.state["initialized"] = False

    def _gather_flat_grad(self):
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.detach().reshape(-1))
        return torch.cat(grads) if grads else torch.tensor([], dtype=torch.float32)

    def _apply_hessian(self, vec):
        # Compute H*vec using the limited-memory SR1 approximation:
        Hv = self.init_hessian_diag * vec
        for s_i, u_i in zip(self.state["s_history"], self.state["u_history"]):
            denom = torch.dot(u_i, s_i)
            if abs(denom.item()) > 1e-12:
                Hv = Hv + (torch.dot(u_i, vec) / denom) * u_i
        return Hv

    def _solve_tr_subproblem(self, g, delta):
        # Solve: minimize 0.5 * s^T B s + g^T s  s.t. ||s|| <= delta.
        # Here, B is the Hessian approximation (via LSR1) and delta is the current trust-region radius.
        device = g.device
        n = g.numel()
        s = torch.zeros_like(g)
        r = g.clone()  # r = g + B*s, with s initially 0 -> r = g.
        p = -r.clone()  # initial search direction
        g_norm = torch.norm(g)
        # Use a capped conjugate gradient method (max iterations set to min(n, 50)):
        for _ in range(min(n, 50)):
            if torch.norm(r) <= 1e-8 * g_norm:
                break
            Bp = self._apply_hessian(p)
            pBp = torch.dot(p, Bp)
            if pBp <= 1e-12:  # negative curvature detected
                sp = torch.dot(s, p)
                pp = torch.dot(p, p)
                A = float(pp)
                B_val = 2 * float(sp)
                C = float(torch.dot(s, s) - delta**2)
                if A < 1e-12:
                    break
                disc = B_val**2 - 4 * A * C
                disc = max(disc, 0.0)
                alpha = (
                    (-B_val + math.sqrt(disc)) / (2 * A)
                    if B_val < 0
                    else (-B_val - math.sqrt(disc)) / (2 * A)
                )
                alpha = max(alpha, 0.0)
                s.add_(p, alpha=alpha)
                break
            alpha = torch.dot(r, r) / pBp
            s_next = s + alpha * p
            if torch.norm(s_next) >= delta:
                sp = torch.dot(s, p)
                pp = torch.dot(p, p)
                A = float(pp)
                B_val = 2 * float(sp)
                C = float(torch.dot(s, s) - delta**2)
                disc = B_val**2 - 4 * A * C
                disc = max(disc, 0.0)
                sqrt_disc = math.sqrt(disc)
                alpha1 = (-B_val + sqrt_disc) / (2 * A) if A > 1e-12 else 0.0
                alpha2 = (-B_val - sqrt_disc) / (2 * A) if A > 1e-12 else 0.0
                alphas = [a for a in (alpha1, alpha2) if a > 1e-12]
                alpha_boundary = min(alphas) if alphas else 0.0
                s.add_(p, alpha=alpha_boundary)
                break
            s = s_next
            r_new = r + alpha * Bp
            if torch.norm(r_new) <= 1e-8 * g_norm:
                r = r_new
                break
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = -r_new + beta * p
            r = r_new
        # Predicted reduction in the quadratic model:
        gTs = float(torch.dot(g, s))
        sBs = float(torch.dot(s, self._apply_hessian(s)))
        predicted_reduction = -(gTs + 0.5 * sBs)
        return s, predicted_reduction

    def _apply_update(self, s):
        offset = 0
        for p in self.param_list:
            if p.numel() == 0:
                continue
            numel = p.numel()
            p.data.add_(s[offset : offset + numel].view_as(p.data))
            offset += numel

    def step(self, closure, old_loss=None, grad=None):
        """
        Performs one optimization step.

        Arguments:
            closure (callable): A function that re-evaluates the model and returns the loss.
            old_loss (float, optional): Precomputed loss. If None, computed via closure.
            grad (Tensor, optional): Precomputed gradient (flattened). If None, gathered from parameters.
        """
        if old_loss is None:
            loss = closure(compute_grad=True)
            loss = loss.item() if type(loss) is not float else loss
            old_loss = float(loss) if loss is not None else None
        if grad is None:
            if not self.model_has_grad:
                grad = torch.cat(
                    [
                        p.grad.detach().view(-1)
                        for p in self.param_list
                        if p.grad is not None
                    ]
                )
            else:
                grad = self.model.grad()

        if grad.numel() == 0:
            # print(f"Stopping TrustRegion algorithm due to ||g||=0.")
            return old_loss

        # Initialize state on first call.
        if not self.state["initialized"]:
            self.state["prev_loss"] = old_loss
            self.state["prev_grad"] = grad.clone()
            params_flat = [p.data.detach().view(-1) for p in self.param_list]
            self.state["prev_params"] = (
                torch.cat(params_flat)
                if params_flat
                else torch.tensor([], device=grad.device)
            )
            self.state["initialized"] = True

        # Solve the trust-region subproblem using the LSR1 Hessian approximation.
        s, predicted_reduction = self._solve_tr_subproblem(grad, self.lr)

        # Save current parameters (flattened) to allow rollback.
        params_flat_current = [p.data.detach().view(-1) for p in self.param_list]
        params_flat_current = (
            torch.cat(params_flat_current)
            if params_flat_current
            else torch.tensor([], device=grad.device)
        )

        # Apply candidate step.
        self._apply_update(s)

        # Evaluate new loss (and compute new gradient).
        new_loss = closure(compute_grad=True)
        device = grad.device
        if not torch.is_tensor(new_loss):
            new_loss = torch.tensor(new_loss, device=device)
        else:
            new_loss = new_loss.to(device)

        new_loss_val = float(new_loss.item()) if new_loss is not None else None
        if not self.model_has_grad:
            new_grad = torch.cat(
                [
                    p.grad.detach().view(-1)
                    for p in self.param_list
                    if p.grad is not None
                ]
            )
        else:
            new_grad = self.model.grad()

        # Compute actual reduction and ratio.
        actual_reduction = (
            old_loss - new_loss_val
            if (old_loss is not None and new_loss_val is not None)
            else None
        )
        rho = (
            actual_reduction / (predicted_reduction + 1e-12)
            if (actual_reduction is not None and predicted_reduction != 0)
            else float("inf")
        )

        # Adjust the trust-region radius (lr) and decide acceptance.
        if (actual_reduction is not None and actual_reduction > 0) and (
            rho >= self.nu_1
        ):
            # Accept step.
            if rho >= self.nu_2 and torch.norm(s) >= self.lr * 0.9:
                self.lr = min(self.max_lr, self.inc_factor * self.lr)
        else:
            # Reject step: roll back.
            offset = 0
            for p in self.param_list:
                if p.numel() == 0:
                    continue
                numel = p.numel()
                p.data.copy_(
                    params_flat_current[offset : offset + numel].view_as(p.data)
                )
                offset += numel
            self.lr = max(self.min_lr, self.dec_factor * self.lr)
            return old_loss

        # Update stored state.
        self.state["prev_loss"] = new_loss_val
        self.state["prev_grad"] = new_grad.clone()
        params_flat_new = [p.data.detach().view(-1) for p in self.param_list]
        self.state["prev_params"] = (
            torch.cat(params_flat_new)
            if params_flat_new
            else torch.tensor([], device=grad.device)
        )

        # LSR1 update: use s (parameter step) and y = new_grad - grad.
        s_vec = s.clone()
        y_vec = new_grad - grad
        B_s = self.init_hessian_diag * s_vec
        u_vec = y_vec - B_s
        denom = torch.dot(u_vec, s_vec)
        if (
            abs(denom.item())
            > 1e-8 * torch.norm(u_vec).item() * torch.norm(s_vec).item()
        ):
            # Limit memory size to max_iter.
            if len(self.state["s_history"]) >= self.max_iter:
                self.state["s_history"].pop(0)
                self.state["u_history"].pop(0)
            self.state["s_history"].append(s_vec.detach().clone())
            self.state["u_history"].append(u_vec.detach().clone())

        return new_loss
