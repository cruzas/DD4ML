# Based on:
# Brust, Johannes, Jennifer B. Erway, and Roummel F. Marcia. "On solving L-SR1 trust-region subproblems." Computational Optimization and Applications 66 (2017): 245-266.
from __future__ import annotations

from typing import List, Optional

import torch

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor


class LSR1:
    """
    Compact limited-memory SR1 Hessian approximation in the form

        B  =  gamma I  +  Psi M Psi^T     with
        Psi  =  Y - gamma*S
        M^{-1} = D + L + L^T - gamma S^T S

    Only M^{-1} is stored (OBS needs it); M is accessed by solving the
    linear system rather than forming the explicit inverse.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        memory_length: int = 10,
        tol: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype | None = None,
    ):
        self.memory_length = int(memory_length)
        self.tol = float(tol)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float32 if dtype is None else dtype

        # Scalar gamma_0  (updated whenever a new (s,y) pair is accepted)
        self.gamma = torch.tensor(float(gamma), device=self.device, dtype=self.dtype)

        # Memory of curvature pairs
        self._S: List[torch.Tensor] = []  # each tensor is shape (n,)
        self._Y: List[torch.Tensor] = []

        # Workspaces filled by ``precompute"
        self.Psi: torch.Tensor | None = None  # (n, k)
        self.Minv: torch.Tensor | None = None  # (k, k)

    def update_memory(self, s: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a curvature pair  (s = x_{k+1}-x_k,  y = g_{k+1}-g_k)  if it
        satisfies the SR1 curvature condition  |y^T s| >= tol * ||s||*||y||.
        """

        # Ensure v is a vector and on the correct device/dtype
        def _prepare(vec: torch.Tensor) -> torch.Tensor:
            if isinstance(vec, WeightParallelizedTensor):
                vec = vec.detach()
            return vec.to(self.device, self.dtype).flatten()

        s = _prepare(s)
        y = _prepare(y)
        if s.norm() <= self.tol or y.norm() <= self.tol:
            # Reject pair - insufficient information
            return

        # SR1 curvature check
        curvature = y.dot(s)
        if abs(curvature) <= self.tol * (torch.norm(s) * torch.norm(y)):
            # Reject pair - insufficient curvature information
            return

        # Candidate gamma and psi
        gamma_cand = (y.dot(y) / curvature).clamp_min(self.tol)
        psi_cand = y - gamma_cand * s
        if psi_cand.norm() <= self.tol * y.norm():
            # Reject pair - trivial or degenerate Psi
            return
        
        # If we have an existing Psi, check if the new one is linearly dependent
        if len(self._S) > 0:
            S_mat = torch.stack(self._S, dim=1)  # (n, k)
            Y_mat = torch.stack(self._Y, dim=1)  # (n, k)
            Psi_mat = Y_mat - self.gamma * S_mat  # (n, k)
            # Full column QR decomposition of Psi_mat
            self.Q, _ = torch.linalg.qr(Psi_mat, mode='reduced')
            # Project psi_cand onto span(Psi_mat)
            alpha_cand = self.Q.transpose(0,1) @ psi_cand # (k,)
            psi_res = psi_cand - self.Q @ alpha_cand  # (n,)
            if psi_res.norm() <= self.tol * psi_cand.norm():
                # Reject pair - new Psi is linearly dependent on existing Psi
                print("Rejecting pair: new Psi is linearly dependent on existing Psi.")
                return
            
        # Maintain limited memory
        if len(self._S) >= self.memory_length:
            # Remove oldest pair
            self._S.pop(0)
            self._Y.pop(0)

        # Store new pair
        self._S.append(s)
        self._Y.append(y)

        # "Adaptive" gamma: use last pair (positive by curvature check)
        self.gamma = gamma_cand

    def precompute(self) -> None:
        """
        Build Psi and M^{-1} from the stored pairs.
        Must be called after every memory update before OBS is invoked.
        """
        if not self._S:
            # No pairs yet: use multiple of identity (in this case 0)
            self.Psi = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
            self.Minv = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
            return

        # Stack all pairs into matrices
        S = torch.stack(self._S, dim=1)  # (n, k)
        Y = torch.stack(self._Y, dim=1)  # (n, k)
        k = S.shape[1]

        # Psi = Y − gamma*S
        self.Psi = Y - self.gamma * S  # (n, k)

        # Compact SR1   M^{-1} = D + L + L^T − gamma * S^T * S
        SY = S.transpose(0, 1) @ Y  # (k, k)
        D = torch.diag(torch.diag(SY))  # (k, k)
        L = torch.tril(SY, diagonal=-1)  # (k, k)

        self.Minv = (
            D + L + L.transpose(0, 1) - self.gamma * (S.transpose(0, 1) @ S)
        )  # (k, k)

        # Small diagonal regularization if badly conditioned
        eye_k = torch.eye(k, device=self.device, dtype=self.dtype)
        lambda_reg = self.tol * torch.norm(self.Minv, p="fro")
        self.Minv += lambda_reg * eye_k

    def B(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the SR1 Hessian approximation: B*v.
        """
        # Ensure v is a vector and on the correct device/dtype
        if isinstance(v, WeightParallelizedTensor):
            v = v.detach().to(self.device, self.dtype)
        else:
            v = v.to(self.device, self.dtype)

        if self.Psi is None or self.Psi.numel() == 0:
            return self.gamma * v

        # Solve  M^{-1}*x = Psi^T * v (without forming M)
        # (M^{-1} already stored) -> x = (M^{-1})^{-1} Psi^T * v
        rhs = self.Psi.transpose(0, 1) @ v  # (k,)
        x = torch.linalg.solve(self.Minv, rhs)  # (k,)
        return self.gamma * v + self.Psi @ x  # (n,)

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
