# Method originally from:
# "On Solving L-SR1 Trust-Region Subproblems" by Brust et al.
# https://arxiv.org/pdf/1506.07222


import warnings

import numpy as np
import torch

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor

try:
    from scipy import array, dot
except ImportError:
    import numpy as np

    array, dot = np.array, np.dot


class OBS:
    def __init__(self):
        super().__init__()
        self.tol = 1e-6

    def _vec_to_wpt(
        self, vec: torch.Tensor, like: WeightParallelizedTensor
    ) -> WeightParallelizedTensor:
        """Convert flat tensor ``vec`` to a ``WeightParallelizedTensor`` with
        the same sharding as ``like``."""
        shards = []
        offset = 0
        for t in like.tensor:
            n = t.numel()
            shards.append(vec[offset : offset + n].view_as(t))
            offset += n
        return WeightParallelizedTensor(
            shards, like.backend, like.master_group, like.rank
        )

    def _clip_to_region(self, p, delta):
        """Guarantee the trust-region constraint on a boundary step.

        The Newton iteration on phi_bar stops at |phi_bar| <= tol, which pins
        ||p|| only to about delta*tol, so the boundary step can land marginally
        outside the region. Trust-region acceptance assumes a feasible step, so
        scale it back when that happens.
        """
        pnorm = torch.sqrt(p.dot(p))
        if float(pnorm) > float(delta) > 0.0:
            return p * (delta / pnorm)
        return p

    def _scaled_identity_step(self, g, delta, gamma):
        """Exact trust-region step when B is a multiple of the identity.

        Reached when Psi spans nothing usable, so B = gamma*I. The subproblem
        min gᵀp + 0.5*gamma*||p||² over ||p|| <= delta then has the closed form
        below, with the boundary step taken whenever gamma <= 0 (the model is
        not convex) or the Newton point falls outside the region.
        """
        gnorm = torch.sqrt(g.dot(g))
        if float(gnorm) == 0.0:
            return g * 0.0
        if float(gamma) > 0.0 and float(gnorm / gamma) <= float(delta):
            return -g / gamma
        return -(delta / gnorm) * g

    @staticmethod
    def _solve_possibly_singular(A, b):
        """Solve A x = b, falling back to the pseudo-inverse.

        A inherits the rank deficiency of Psi, so an exact solve is not always
        available; a least-squares solution is the right answer there.
        """
        try:
            return torch.linalg.solve(A, b)
        except Exception:
            return torch.linalg.pinv(A) @ b

    def solve_tr_subproblem(self, g, delta, gamma, Psi, Minv):
        # Check that g, delta, gamma, Psi, and Minv do not have NaN or Inf values
        if torch.isnan(g).any() or torch.isinf(g).any():
            raise ValueError(f"Gradient g contains NaN or Inf values. g: {g}")
        if torch.isnan(delta) or torch.isinf(delta):
            raise ValueError(f"Delta contains NaN or Inf values. Delta: {delta}")
        if torch.isnan(gamma) or torch.isinf(gamma):
            raise ValueError(f"Gamma contains NaN or Inf values. Gamma: {gamma}")
        if torch.isnan(Psi).any() or torch.isinf(Psi).any():
            raise ValueError(f"Psi contains NaN or Inf values. Psi: {Psi}")
        if torch.isnan(Minv).any() or torch.isinf(Minv).any():
            raise ValueError(f"Minv contains NaN or Inf values. Minv: {Minv}")
        if delta < 0:
            raise ValueError(f"Delta must be non-negative. Delta: {delta}")

        PsiPsi = torch.matmul(Psi.transpose(0, 1), Psi)
        PsiPsi = (PsiPsi + PsiPsi.transpose(0, 1)) / 2.0

        # Psi = Y - gamma*S loses column rank in two ordinary situations: when
        # the memory holds more pairs than the problem has dimensions, and when
        # the iterates stop varying so that successive pairs become near
        # parallel. Psi^T Psi then has no Cholesky factor -- and it squares the
        # condition number of Psi, so it fails well before Psi itself is
        # numerically singular. That is what used to abort the entire
        # second-order path.
        #
        # Use a rank-revealing eigendecomposition instead. With
        # Psi^T Psi = V Sigma V^T and (Sigma_r, V_r) its numerically nonzero
        # part, C = V_r Sigma_r^{-1/2} makes Q = Psi C an orthonormal basis of
        # range(Psi), and R_r = Sigma_r^{1/2} V_r^T takes over the role of the
        # Cholesky factor, since Q^T Psi = R_r. The compact representation
        # B = gamma I + Psi M Psi^T is untouched; only the factorisation used to
        # diagonalise it changes. No curvature information is discarded -- the
        # directions dropped are exactly those Psi does not span, on which B
        # already acts as gamma I.
        sigma, V = torch.linalg.eigh(PsiPsi)
        sigma = torch.clamp(sigma, min=0.0)  # PSD up to rounding
        sigma_max = sigma.max() if sigma.numel() else sigma.new_zeros(())

        if sigma.numel() == 0 or float(sigma_max) <= 0.0:
            # Psi spans nothing usable, so B is gamma*I everywhere.
            return self._scaled_identity_step(g, delta, gamma)

        rank_tol = sigma_max * max(Psi.shape) * torch.finfo(Psi.dtype).eps
        keep = sigma > rank_tol
        if not bool(keep.any()):
            return self._scaled_identity_step(g, delta, gamma)

        Sigma_r = sigma[keep]
        V_r = V[:, keep]
        sqrt_Sigma_r = torch.sqrt(Sigma_r)
        C = V_r / sqrt_Sigma_r  # (k, r) = V_r Sigma_r^{-1/2}
        R_r = sqrt_Sigma_r.unsqueeze(1) * V_r.transpose(0, 1)  # (r, k)

        MR = torch.linalg.solve(Minv, R_r.transpose(0, 1))  # (k, r)
        RMR = torch.matmul(R_r, MR)  # (r, r)
        RMR = (RMR + RMR.transpose(0, 1)) / 2.0  # this forces eigvenvalues to be real

        D, U = torch.linalg.eigh(RMR)
        sorted_indices = torch.argsort(D)
        D = D[sorted_indices]
        U = U[:, sorted_indices]

        sizeD = D.shape[0]  # the numerical rank r, not the memory length k
        Lambda_one = D + gamma
        Lambda = torch.cat((Lambda_one, gamma.reshape(1)))
        Lambda = torch.where(
            torch.abs(Lambda) < self.tol, torch.zeros_like(Lambda), Lambda
        )
        lambda_min = torch.minimum(Lambda[0], gamma.reshape(()))

        RinvU = torch.matmul(C, U)  # (k, r), replaces solve(R, U)

        P_parallel = torch.matmul(Psi, RinvU)
        # Ensure Psi and g have compatible dtypes for matrix multiplication
        g_compatible = g.to(dtype=Psi.dtype)
        PsiTg = torch.matmul(Psi.transpose(0, 1), g_compatible)
        g_parallel = torch.matmul(RinvU.transpose(0, 1), PsiTg)

        gg = g.dot(g)
        gpgp = g_parallel.dot(g_parallel)

        diff = gg - gpgp
        a_kp2 = torch.sqrt(torch.clamp(diff, min=0.0))

        a_j = torch.cat((g_parallel, a_kp2.view(-1)))
        # a_j / Lambda, but the line above deliberately zeroes tiny eigenvalues.
        # Treat 0/0 as 0 and x/0 with x != 0 as infinity, so the interior test
        # below falls through to the boundary case instead of comparing against
        # a NaN -- which is how this used to reach Newton with unusable data.
        nonzero = torch.abs(Lambda) > 0
        helpp = torch.zeros_like(a_j)
        helpp[nonzero] = a_j[nonzero] / Lambda[nonzero]
        blown = (~nonzero) & (torch.abs(a_j) > 0)
        helpp[blown] = float("inf")

        if lambda_min > 0 and torch.norm(helpp) <= delta:
            pStar = self.ComputeSBySMW(gamma, g_compatible, PsiTg, Psi, Minv, PsiPsi)
            return pStar
        elif lambda_min <= 0 and self.phiBar_f(-lambda_min, Lambda, a_j, delta) >= 0:
            sigmaStar = -lambda_min
            v = torch.zeros(sizeD + 1, dtype=a_j.dtype, device=a_j.device)
            idx_pseudo = torch.where(torch.abs(Lambda + sigmaStar) > self.tol)
            v[idx_pseudo] = a_j[idx_pseudo] / (Lambda[idx_pseudo] + sigmaStar)

            if torch.abs(gamma + sigmaStar) < self.tol:
                pStar = -1.0 * torch.matmul(P_parallel, v[:sizeD])
            else:
                term1 = -1.0 * torch.matmul(P_parallel, v[:sizeD])
                # PsiPsi is singular exactly when Psi is rank deficient,
                # so apply its pseudo-inverse via the factors above.
                term_help = V_r @ ((V_r.transpose(0, 1) @ PsiTg) / Sigma_r)
                term2 = 1.0 / (gamma + sigmaStar) * torch.matmul(Psi, term_help)
                if isinstance(g, WeightParallelizedTensor):
                    term3 = g.div(gamma + sigmaStar)
                    term1 = self._vec_to_wpt(term1, g)
                    term2 = self._vec_to_wpt(term2, g)
                    pStar = term1 + term2 - term3
                else:
                    term3 = g / (gamma + sigmaStar)
                    pStar = term1 + term2 - term3

            if lambda_min < 0:
                alpha_sq = delta**2 - pStar.dot(pStar)
                alpha = torch.sqrt(torch.clamp(alpha_sq, min=0.0))
                pHatStar = pStar

                if torch.abs(lambda_min - Lambda[0]) < self.tol:
                    zstar = (
                        (1.0 / torch.norm(P_parallel[:, 0])) * alpha * P_parallel[:, 0]
                    )
                else:
                    e = torch.zeros_like(g)
                    found = False

                    for i in range(sizeD):
                        e[i] = 1
                        u_min = e - torch.matmul(
                            P_parallel, P_parallel[i, :].transpose(0, 1)
                        )
                        if torch.norm(u_min) > self.tol:
                            found = True
                            break

                        e[i] = 0

                    if not found:
                        e[sizeD] = 1
                        u_min = e - torch.matmul(
                            P_parallel, P_parallel[sizeD, :].transpose(0, 1)
                        )

                    u_min = u_min / torch.norm(u_min)
                    zstar = alpha * u_min

                pStar = pHatStar + zstar

            return self._clip_to_region(pStar, delta)
        else:
            if lambda_min > 0:
                sigmaStar = self.Newton(0, Lambda, a_j, delta)
            else:
                sigmaHat = torch.max(a_j / delta - Lambda)
                if sigmaHat > -lambda_min:
                    sigmaStar = self.Newton(sigmaHat, Lambda, a_j, delta)
                else:
                    sigmaStar = self.Newton(-lambda_min, Lambda, a_j, delta)

            if torch.isnan(sigmaStar) or torch.isinf(sigmaStar):
                sigmaStar = self.Newton(0, Lambda, a_j, delta)

            pStar = self.ComputeSBySMW(
                gamma + sigmaStar, g_compatible, PsiTg, Psi, Minv, PsiPsi
            )
            return self._clip_to_region(pStar, delta)

    def ComputeSBySMW(self, tauStar, g, PsiTg, Psi, Minv, PsiPsi):
        """p = -(B + sigma I)^-1 g  with  B + sigma I = tau I + Psi M Psi^T.

        By Sherman-Morrison-Woodbury, with tau = tauStar and Minv = M^-1,

            (tau I + Psi M Psi^T)^-1
                = (1/tau) I - (1/tau) Psi (tau Minv + Psi^T Psi)^-1 Psi^T

        so both terms carry the 1/tau factor. The update term used to be
        applied without it, which left every step this routine produced wrong
        by a factor of tau -- and since both the interior branch and the final
        boundary branch return through here, that was most of the algorithm.
        """
        W = tauStar * Minv + PsiPsi
        WinvPsiTg = self._solve_possibly_singular(W, PsiTg)
        update = (Psi @ WinvPsiTg) / tauStar
        if isinstance(g, WeightParallelizedTensor):
            update = self._vec_to_wpt(update, g)
            return (-1.0 / tauStar) * g + update  # pstar
        return (-1.0 / tauStar) * g + update  # pstar

    def _phi_bar_terms(self, sigma, Dd, a_j):
        """Shared setup for phiBar_f / phiBar_fg.

        phi_bar(sigma) = 1/||p(sigma)|| - 1/delta with p_i = -a_i/(lambda_i+sigma).

        A component with a_i = 0 contributes nothing to ||p|| and must simply be
        skipped. Only a genuine pole -- lambda_i + sigma = 0 while a_i != 0,
        where ||p|| blows up -- makes phi_bar saturate at -1/delta.

        The guard here used to fire whenever *any* a_i or lambda_i + sigma was
        small. That is the common case rather than the exceptional one: the
        direction orthogonal to range(Psi) carries a_i = 0 whenever Psi spans
        the whole space. Newton was therefore handed a constant function, its
        derivative sentinel 1/tol, and returned a shift of essentially zero --
        so the boundary branch handed back an unconstrained Newton step that
        violated the trust region.
        """
        D = Dd + sigma
        near_zero = torch.abs(D) < self.tol
        pole = near_zero & (torch.abs(a_j) > self.tol)
        return D, near_zero, bool(pole.any())

    def phiBar_f(self, sigma, Dd, a_j, delta):
        D, near_zero, has_pole = self._phi_bar_terms(sigma, Dd, a_j)
        if has_pole:
            return -1.0 / delta

        safe = ~near_zero
        pnorm2 = torch.sum((a_j[safe] / D[safe]) ** 2)
        normP = torch.sqrt(pnorm2)
        if float(normP) == 0.0:
            # p(sigma) = 0, which is strictly inside any positive radius.
            return torch.full_like(pnorm2, float("inf"))
        phiBar = 1.0 / normP - 1.0 / delta
        return phiBar

    def Newton(self, x0, Lambda, a_j, delta):
        maxIter = 200

        x = x0
        k = 0

        f, g = self.phiBar_fg(x, Lambda, a_j, delta)

        while torch.abs(f) > self.tol and k < maxIter:
            x = x - f / g
            f, g = self.phiBar_fg(x, Lambda, a_j, delta)
            k = k + 1

        if torch.isnan(x) or torch.isinf(x):
            warnings.warn(
                "OBS: the phiBar Newton iteration produced a non-finite root; "
                "the returned trust-region shift is unusable.",
                RuntimeWarning,
                stacklevel=2,
            )

        return x

    def phiBar_fg(self, sigma, Dd, a_j, delta):
        """phiBar_f together with its derivative; see _phi_bar_terms."""
        D, near_zero, has_pole = self._phi_bar_terms(sigma, Dd, a_j)
        if has_pole:
            phiBar = -torch.ones_like(delta) / delta
            phiBar_g = torch.ones_like(delta) / self.tol
            return phiBar, phiBar_g

        safe = ~near_zero
        p = a_j[safe] / D[safe]
        normP = torch.norm(p)
        if float(normP) == 0.0:
            phiBar = torch.full_like(delta, float("inf"))
            phiBar_g = torch.ones_like(delta) / self.tol
            return phiBar, phiBar_g

        phiBar = 1 / normP - 1 / delta

        phiBar_g = torch.sum((a_j[safe] ** 2) / (D[safe] ** 3))
        phiBar_g = phiBar_g / (normP**3)

        return phiBar, phiBar_g
