# Limited-memory SR1 “compact” Hessian approximation for use with the
# OBS trust-region sub-problem solver (Fletcher & Gould, 2021).
#
# The class maintains the triples (γ, Ψ, Minv) required by equation (11) in the above work
from __future__ import annotations

from typing import List, Optional

import torch

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor


class LSR1:
    """
    Compact limited-memory SR1 Hessian approximation in the form

        B  =  gamma I  +  Ψ M Ψᵀ     with
        Ψ  =  Y - gamma S
        M⁻¹ = D + L + Lᵀ - gamma SᵀS        (eq. (9) in the OBS paper)

    Only M⁻¹ is stored (OBS needs it); M is accessed by solving the
    linear system rather than forming the explicit inverse.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        memory_length: int = 10,
        tol: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype | None = None,
        # TODO: add norm_type
    ):
        self.memory_length = int(memory_length)
        self.tol = float(tol)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float32 if dtype is None else dtype

        # scalar γ₀  (updated whenever a new (s,y) pair is accepted)
        self.gamma = torch.tensor(float(gamma), device=self.device, dtype=self.dtype)

        # memory of curvature pairs
        self._S: List[torch.Tensor] = []  # each tensor is shape (n,)
        self._Y: List[torch.Tensor] = []

        # workspaces filled by `precompute`
        self.Psi: torch.Tensor | None = None  # (n, k)
        self.Minv: torch.Tensor | None = None  # (k, k)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def update_memory(self, s: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a curvature pair  (s = x_{k+1}-x_k,  y = ∇f_{k+1}-∇f_k)  if it
        satisfies the SR1 curvature condition  |yᵀs| ≥ ε‖s‖‖y‖.
        """

        def _prepare(vec: torch.Tensor) -> torch.Tensor:
            if isinstance(vec, WeightParallelizedTensor):
                vec = vec.detach()
            return vec.to(self.device, self.dtype).flatten()

        s = _prepare(s)
        y = _prepare(y)
        if s.norm() <= self.tol or y.norm() <= self.tol:
            return

        curvature = y.dot(s)
        if abs(curvature) <= self.tol * (
            torch.norm(s) * torch.norm(y)
        ):  # <= in case of 0 compared against 0
            # reject pair – insufficient curvature information
            return

        # maintain limited memory
        if len(self._S) >= self.memory_length:
            self._S.pop(0)
            self._Y.pop(0)

        self._S.append(s)
        self._Y.append(y)

        # “adaptive” γ: use last pair  γ = (yᵀy)/(yᵀs)  (positive by curvature check)
        self.gamma = (y.dot(y) / curvature).clamp_min(self.tol)

    def precompute(self) -> None:
        """
        Build Ψ and M⁻¹ from the stored pairs.  Must be called after every
        memory update **before** OBS is invoked.
        """
        if not self._S:
            # no pairs yet: use pure multiple of identity
            self.Psi = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
            self.Minv = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
            return

        S = torch.stack(self._S, dim=1)  # (n, k)
        Y = torch.stack(self._Y, dim=1)  # (n, k)
        k = S.shape[1]

        # Ψ = Y − γ S
        self.Psi = Y - self.gamma * S  # (n, k)

        # Compact SR1   M⁻¹ = D + L + Lᵀ − γ SᵀS
        SY = S.transpose(0, 1) @ Y  # (k, k)
        D = torch.diag(torch.diag(SY))  # (k, k)
        L = torch.tril(SY, diagonal=-1)  # (k, k)

        self.Minv = (
            D + L + L.transpose(0, 1) - self.gamma * (S.transpose(0, 1) @ S)
        )  # (k, k)

        # Small diagonal regularisation if badly conditioned
        eye_k = torch.eye(k, device=self.device, dtype=self.dtype)
        lambda_reg = self.tol * torch.norm(self.Minv, p="fro")
        self.Minv += lambda_reg * eye_k

    def B(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the SR1 Hessian approximation:  B v.
        """
        if isinstance(v, WeightParallelizedTensor):
            v = v.detach().to(self.device, self.dtype)
        else:
            v = v.to(self.device, self.dtype)
        if self.Psi is None or self.Psi.numel() == 0:
            return self.gamma * v

        # Solve  M x = Ψᵀ v   without forming M
        #       (M⁻¹ already stored)  =>  x = (M⁻¹)⁻¹ Ψᵀ v
        rhs = self.Psi.transpose(0, 1) @ v  # (k,)
        # x = (Minv)⁻¹ rhs   ==>   solve instead of inverse
        x = torch.linalg.solve(self.Minv, rhs)  # (k,)
        return self.gamma * v + self.Psi @ x  # (n,)

    # convenience accessors -------------------------------------------------
    @property
    def S(self) -> torch.Tensor:
        return (
            torch.stack(self._S, dim=1)
            if self._S
            else torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        )

    @property
    def Y(self) -> torch.Tensor:
        return (
            torch.stack(self._Y, dim=1)
            if self._Y
            else torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        )
