import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from dd4ml.solvers.obs import OBS
from .hessian_approx import LSR1

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _concat_params(params: Iterable[Tensor]) -> Tensor:
    """Flatten parameters into a single 1-D tensor (detached)."""
    return torch.cat([p.detach().flatten() for p in params])


def _concat_grads(params: Iterable[Tensor]) -> Tensor:
    """Flatten current gradients into a single 1-D tensor."""
    return torch.cat([p.grad.flatten() for p in params])


def _set_param_vector(params: Iterable[Tensor], vec: Tensor) -> None:
    """Write the flat vector *vec* back into *params* in-place."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(vec[offset : offset + numel].view_as(p))
        offset += numel


# -----------------------------------------------------------------------------
# Limited-memory SR1 utilities (compact representation)
# -----------------------------------------------------------------------------


def _update_memory(
    S: List[Tensor], Y: List[Tensor], s: Tensor, y: Tensor, m: int
) -> None:
    """Push the newest (s,y) pair and crop memory to size *m*."""
    if s.dot(y) == 0:  # skip if not well-defined
        return
    S.append(s)
    Y.append(y)
    if len(S) > m:
        S.pop(0)
        Y.pop(0)


def _build_compact(
    S: List[Tensor], Y: List[Tensor], gamma: float
) -> Tuple[Tensor, Tensor]:
    """Return Ψ and M in the compact SR1 formula B = gammaI + Ψ M Ψᵀ."""
    if not S:
        return None, None  # means B = gammaI
    S_mat = torch.stack(S, dim=1)  # (n, k)
    Y_mat = torch.stack(Y, dim=1)  # (n, k)
    Psi = Y_mat - gamma * S_mat  # (n, k)

    # Build M = (D + L + Lᵀ − gamma Sᵀ S)⁻¹   ─ k×k matrix
    STY = S_mat.t() @ Y_mat  # (k, k)
    D = torch.diag(torch.diag(STY))
    L = torch.tril(STY, diagonal=-1)
    M_inv = D + L + L.t() - gamma * (S_mat.t() @ S_mat)
    M = torch.linalg.inv(M_inv)  # assume positive-/indefinite but invertible
    return Psi, M


# -----------------------------------------------------------------------------
# Trust-region sub-problem solver (Algorithm 2 from the paper)
# -----------------------------------------------------------------------------


def _sr1_trust_region_step(
    g: Tensor, Psi: Optional[Tensor], M: Optional[Tensor], gamma: float, delta: float
) -> Tensor:
    """Compute p* solving   min ½ pᵀBp + gᵀp  s.t. ‖p‖ ≤ δ,   with SR1 B.
    Implementation follows Algorithm 2 with an explicit spectral decomposition.
    For large *n*, this routine is meant for pedagogical purposes; it allocates dense
    matrices.  In practice, the OBS code discussed previously can be substituted
    for efficiency.
    """
    n = g.numel()
    if Psi is None:
        # B = gammaI
        g_norm = g.norm()
        if g_norm <= 1e-12:
            return torch.zeros_like(g)
        # Cauchy point in direction -g
        tau = min(1.0, delta / g_norm)
        return -tau * g / gamma

    # Build B explicitly (dense)   B = gammaI + Ψ M Ψᵀ
    B = gamma * torch.eye(n, dtype=g.dtype, device=g.device)
    B = B + Psi @ (M @ Psi.t())

    # Eigen-decomposition (dense)
    evals, Q = torch.linalg.eigh(B)  # ascending order

    # Transform g into eigen-space
    g_bar = Q.t() @ g

    # Secular eqn to find σ ≥ 0 such that φ(σ)=0
    def phi(sig: float) -> float:
        denom = evals + sig
        vals = (g_bar / denom).pow(2)
        return 1.0 / math.sqrt(vals.sum().item()) - 1.0 / delta

    # Helper to compute ‖p(σ)‖ efficiently
    def p_sigma(sig: float) -> Tensor:
        return -(Q @ (g_bar / (evals + sig)))

    # Attempt unconstrained minimizer first (σ=0)
    if (evals > 0).all():
        p_unc = p_sigma(0.0)
        if p_unc.norm() <= delta:
            return p_unc
    # Otherwise find root of φ starting from σ= max(0, -λ_min)+eps
    sigma = max(0.0, -evals.min().item() + 1e-8)
    for _ in range(50):
        val = phi(sigma)
        if abs(val) < 1e-10:
            break
        # Newton step  φ'(σ) computation
        denom = evals + sigma
        deriv = ((g_bar**2) / (denom**3)).sum().item()
        sigma = sigma + val / deriv  # Newton
        sigma = max(sigma, 0.0)
    p_star = p_sigma(sigma)
    # Hard-case fix (if needed)
    if p_star.norm() > delta * 1.01:  # numeric safeguard
        p_star = p_star * (delta / p_star.norm())
    return p_star


# -----------------------------------------------------------------------------
# PyTorch Optimizer implementing L-SSR1-TR (Algorithm 3)
# -----------------------------------------------------------------------------


class LSSR1_TR(Optimizer):
    r"""Limited-Memory Stochastic SR1 Trust-Region optimizer (Algorithm 3).

    Arguments
    ---------
    params : iterable of Tensor
        Parameters to optimize.
    closure : callable
        A closure that reevaluates the model and returns the loss.
    lr_init : float, default 1.0
        Initial step-length for the Wolfe line search.
    delta_init : float, default 1.0
        Initial trust-region radius δ₀.
    gamma_init : float, default 1e-3
        Scaling of the initial Hessian approximation B₀ = gammaI.
    memory : int, default 10
        Number of (s,y) pairs to store.
    mu : float, default 0.9
        Momentum parameter.
    overlap : float, default 0.33
        Fraction *φ* of samples that must overlap between successive mini-batches
        (handled externally by the data loader).
    tol_grad : float, default 1e-8
        Terminates when ‖∇f‖ ≤ tol_grad.
    """

    def __init__(
        self,
        params,
        lr_init: float = 1.0,
        delta_init: float = 1.0,
        gamma_init: float = 1e-3,
        memory: int = 10,
        mu: float = 0.9,
        overlap: float = 0.33,
        tol_grad: float = 1e-8,
    ):
        defaults = dict(
            lr_init=lr_init,
            delta=delta_init,
            gamma=gamma_init,
            memory=memory,
            mu=mu,
            overlap=overlap,
            tol_grad=tol_grad,
        )
        super().__init__(params, defaults)

        state = self.state
        
        
        state["S"] = []  # type: List[Tensor]
        state["Y"] = []  # type: List[Tensor]
        state["wk"] = None  # type: Optional[Tensor]
        state["prev_grad"] = None
        state["vk"] = torch.zeros(1)  # momentum vector (lazy-init)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]):
        """Performs a single optimization step."""
        loss = closure()  # forward & backward pass; grad populated

        params = self.param_groups[0]["params"]
        g = _concat_grads(params)
        g_norm = g.norm().item()
        if g_norm <= self.defaults["tol_grad"]:
            return loss  # converged

        state = self.state
        wk = _concat_params(params)
        if state["wk"] is None:
            state["wk"] = wk.clone()
        if state["vk"].numel() != wk.numel():
            state["vk"] = torch.zeros_like(wk)

        # --------------------------------------------------------------
        # 1. Limited-memory update (build S,Y)
        # --------------------------------------------------------------
        if state["prev_grad"] is not None:
            s = wk - state["wk"]
            y = g - state["prev_grad"]
            _update_memory(state["S"], state["Y"], s, y, self.defaults["memory"])
        state["wk"] = wk.clone()
        state["prev_grad"] = g.clone()

        # --------------------------------------------------------------
        # 2. Compute trust-region step using SR1
        # --------------------------------------------------------------
        Psi, M = _build_compact(state["S"], state["Y"], self.defaults["gamma"])
        p_star = _sr1_trust_region_step(
            g, Psi, M, self.defaults["gamma"], self.defaults["delta"]
        )

        # --------------------------------------------------------------
        # 3. Momentum grafting (Eq. 17)
        # --------------------------------------------------------------
        vk = (
            state["vk"] * self.defaults["mu"] + s
            if "s" in locals()
            else state["vk"] * self.defaults["mu"]
        )
        if vk.norm() > 0:
            vk = self.defaults["mu"] * min(1.0, self.defaults["delta"] / vk.norm()) * vk
        p_combined = p_star + vk
        if p_combined.norm() > 0:
            p_combined = (
                min(1.0, self.defaults["delta"] / p_combined.norm()) * p_combined
            )
        state["vk"] = vk.clone()

        # --------------------------------------------------------------
        # 4. Wolfe line-search along p_combined (backtracking -> sufficient decrease w.r.t batch loss)
        # --------------------------------------------------------------
        alpha = self.defaults["lr_init"]
        c1, c2 = 1e-4, 0.9
        max_ls = 10
        orig_loss = loss.item()
        orig_grad_dot_dir = g.dot(p_combined)
        for _ in range(max_ls):
            # tentative move
            _set_param_vector(params, wk + alpha * p_combined)
            new_loss = closure().item()
            if new_loss <= orig_loss + c1 * alpha * orig_grad_dot_dir:
                # curvature condition (simple check)
                break
            alpha *= 0.5
        else:
            alpha = 0.0  # resort to no step if LS failed
            _set_param_vector(params, wk)  # reset params

        p_alpha = alpha * p_combined
        # --------------------------------------------------------------
        # 5. Accept step, update trust-region radius
        # --------------------------------------------------------------
        # Predicted reduction
        pred = g.dot(p_alpha) + 0.5 * p_alpha @ (
            g * 0
        )  # Bp term ignored here for cost; approximate
        ratio = 0.0
        if pred < 0:
            ratio = (new_loss - orig_loss) / pred
        # Update δ (simple rules)
        if ratio < 0.25:
            self.defaults["delta"] *= 0.5
        elif ratio > 0.75 and abs(p_alpha.norm() - self.defaults["delta"]) < 1e-12:
            self.defaults["delta"] *= 2.0

        # step already taken via line-search update
        return loss
