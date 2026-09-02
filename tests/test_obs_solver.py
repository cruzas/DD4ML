"""Correctness tests for the OBS trust-region subproblem solver.

Reference:

    J. Brust, J. B. Erway, R. F. Marcia, "On solving L-SR1 trust-region
    subproblems", Computational Optimization and Applications 66:245-266 (2017).
    https://arxiv.org/pdf/1506.07222

OBS minimises the quadratic model

    m(p) = g^T p + 0.5 p^T B p     subject to   ||p|| <= delta

where B is held in the compact L-SR1 form  B = gamma I + Psi M Psi^T, with
Minv = M^-1 supplied by dd4ml.optimizers.lsr1.LSR1.

The core of this module is a comparison against an exact dense solve. B is small
enough here to diagonalise outright, so the true constrained minimiser can be
computed independently and OBS held to it. That is what exposed the missing
1/tau factor in ComputeSBySMW, which had made every step this solver returned
wrong by a factor of tau; a smoke test that only asked "does it run" passed
throughout.
"""

import pytest
import torch

from dd4ml.solvers.obs import OBS

DTYPE = torch.float64


def exact_tr_solution(B, g, delta):
    """Exact minimiser of g^T p + 0.5 p^T B p over ||p|| <= delta.

    Diagonalise B and apply the Moré-Sorensen characterisation: the solution is
    p(sigma) = -Q (Q^T g)/(lambda + sigma) for the unique sigma >= max(0,
    -lambda_min) that puts ||p|| on the boundary, or the unconstrained Newton
    point when that already lies inside. sigma is found by bisection, which is
    slow but unimpeachable at these sizes.
    """
    lam, Q = torch.linalg.eigh(B)
    ghat = Q.T @ g

    def p_of(sigma):
        d = lam + sigma
        out = torch.zeros_like(ghat)
        nonzero = d.abs() > 1e-14
        out[nonzero] = -ghat[nonzero] / d[nonzero]
        return Q @ out

    if lam.min() > 0:
        p = p_of(torch.zeros((), dtype=B.dtype))
        if p.norm() <= delta * (1 + 1e-12):
            return p

    lo = max(0.0, float(-lam.min())) + 1e-14
    hi = lo + 1.0
    while p_of(torch.tensor(hi, dtype=B.dtype)).norm() > delta and hi < 1e18:
        hi *= 2
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if p_of(torch.tensor(mid, dtype=B.dtype)).norm() > delta:
            lo = mid
        else:
            hi = mid
    return p_of(torch.tensor(0.5 * (lo + hi), dtype=B.dtype))


def model_value(B, g, p):
    return float(g @ p + 0.5 * p @ B @ p)


def dense_B(gamma, Psi, Minv):
    B = gamma * torch.eye(Psi.shape[0], dtype=DTYPE) + Psi @ torch.linalg.inv(
        Minv
    ) @ Psi.transpose(0, 1)
    return (B + B.transpose(0, 1)) / 2


def _random_case(seed, n=None, k=None, rank_deficient=False):
    gen = torch.Generator().manual_seed(seed)
    n = n or int(torch.randint(3, 9, (1,), generator=gen))
    k = k or int(torch.randint(1, min(n, 4) + 1, (1,), generator=gen))
    Psi = torch.randn(n, k, generator=gen, dtype=DTYPE)
    if rank_deficient and k > 1:
        # Make the last column an exact copy of the first.
        Psi[:, -1] = Psi[:, 0]
    A = torch.randn(k, k, generator=gen, dtype=DTYPE)
    Minv = A @ A.transpose(0, 1) + 0.5 * torch.eye(k, dtype=DTYPE)
    gamma = torch.rand((), generator=gen, dtype=DTYPE) * 2 + 0.1
    g = torch.randn(n, generator=gen, dtype=DTYPE)
    delta = torch.rand((), generator=gen, dtype=DTYPE) * 2 + 0.1
    return g, delta, gamma, Psi, Minv


def _check_against_exact(g, delta, gamma, Psi, Minv):
    B = dense_B(gamma, Psi, Minv)
    p = OBS().solve_tr_subproblem(g, delta, gamma, Psi, Minv)

    assert torch.isfinite(p).all(), f"non-finite step: {p}"
    assert float(p.norm()) <= float(delta) * (1 + 1e-8), (
        f"step leaves the trust region: ||p|| = {float(p.norm())}, delta = {float(delta)}"
    )

    m_obs = model_value(B, g, p)
    m_ref = model_value(B, g, exact_tr_solution(B, g, delta))
    assert m_obs <= m_ref + 1e-8 * max(1.0, abs(m_ref)), (
        f"model value {m_obs} is worse than the exact minimum {m_ref}"
    )
    return p


@pytest.mark.parametrize("seed", range(40))
def test_matches_the_exact_solution(seed):
    """OBS must attain the true constrained minimum, not merely a descent step."""
    _check_against_exact(*_random_case(seed))


@pytest.mark.parametrize("seed", range(15))
def test_handles_rank_deficient_psi(seed):
    """A duplicated column makes Psi^T Psi singular.

    This is the configuration that used to raise LinAlgError out of
    torch.linalg.cholesky and abort the whole second-order path.
    """
    _check_against_exact(*_random_case(seed, n=6, k=3, rank_deficient=True))


def test_handles_more_pairs_than_dimensions():
    """k > n forces rank deficiency structurally: Psi cannot have k independent
    columns when it only has n rows."""
    _check_against_exact(*_random_case(0, n=2, k=4))


def test_handles_near_parallel_columns():
    """As trust-region iterates converge, successive curvature pairs become
    near-parallel and Psi^T Psi goes numerically singular well before Psi does.
    """
    gen = torch.Generator().manual_seed(3)
    n, k = 8, 4
    base = torch.randn(n, generator=gen, dtype=DTYPE)
    Psi = torch.stack(
        [base + 1e-11 * torch.randn(n, generator=gen, dtype=DTYPE) for _ in range(k)],
        dim=1,
    )
    A = torch.randn(k, k, generator=gen, dtype=DTYPE)
    Minv = A @ A.transpose(0, 1) + 0.5 * torch.eye(k, dtype=DTYPE)
    _check_against_exact(
        torch.randn(n, generator=gen, dtype=DTYPE),
        torch.tensor(0.7, dtype=DTYPE),
        torch.tensor(0.9, dtype=DTYPE),
        Psi,
        Minv,
    )


def test_interior_step_is_the_newton_step():
    """With a large radius the solution is unconstrained: p = -B^-1 g."""
    g, _, gamma, Psi, Minv = _random_case(5, n=5, k=2)
    delta = torch.tensor(1e6, dtype=DTYPE)
    B = dense_B(gamma, Psi, Minv)

    p = OBS().solve_tr_subproblem(g, delta, gamma, Psi, Minv)

    assert torch.allclose(p, torch.linalg.solve(B, -g), atol=1e-9), (
        "interior case must return the Newton step"
    )


def test_boundary_step_lies_on_the_boundary():
    """With a small radius the solution is active: ||p|| == delta."""
    g, _, gamma, Psi, Minv = _random_case(6, n=5, k=2)
    delta = torch.tensor(1e-3, dtype=DTYPE)

    p = OBS().solve_tr_subproblem(g, delta, gamma, Psi, Minv)

    assert float(p.norm()) == pytest.approx(float(delta), rel=1e-6)


def test_smw_inverse_matches_a_dense_solve():
    """ComputeSBySMW must reproduce -(tau I + Psi M Psi^T)^-1 g.

    Pinned directly because the 1/tau factor on the update term was missing:
    both the interior branch and the final boundary branch return through here,
    so the error propagated to essentially every second-order step.
    """
    g, _, gamma, Psi, Minv = _random_case(7, n=6, k=3)
    tau = float(gamma) + 0.37
    PsiPsi = Psi.transpose(0, 1) @ Psi
    PsiTg = Psi.transpose(0, 1) @ g

    p = OBS().ComputeSBySMW(torch.tensor(tau, dtype=DTYPE), g, PsiTg, Psi, Minv, PsiPsi)

    shifted = tau * torch.eye(Psi.shape[0], dtype=DTYPE) + Psi @ torch.linalg.inv(
        Minv
    ) @ Psi.transpose(0, 1)
    assert torch.allclose(p, torch.linalg.solve(shifted, -g), atol=1e-10)


def test_zero_psi_falls_back_to_a_scaled_identity_model():
    """With no curvature information B = gamma*I, which has a closed form."""
    n = 4
    Psi = torch.zeros(n, 2, dtype=DTYPE)
    Minv = torch.eye(2, dtype=DTYPE)
    g = torch.tensor([1.0, -2.0, 0.5, 0.25], dtype=DTYPE)
    gamma = torch.tensor(2.0, dtype=DTYPE)

    # Large radius: the Newton point -g/gamma is interior.
    p = OBS().solve_tr_subproblem(g, torch.tensor(10.0, dtype=DTYPE), gamma, Psi, Minv)
    assert torch.allclose(p, -g / gamma, atol=1e-12)

    # Small radius: the step saturates on the boundary along -g.
    delta = torch.tensor(0.1, dtype=DTYPE)
    p = OBS().solve_tr_subproblem(g, delta, gamma, Psi, Minv)
    assert float(p.norm()) == pytest.approx(float(delta), rel=1e-12)
    assert torch.allclose(p / p.norm(), -g / g.norm(), atol=1e-12)


def test_newton_returns_a_tensor_when_it_exits_immediately():
    """Newton must cope with the initial point already solving phi_bar = 0.

    x0 is passed as a plain 0 from two call sites, and when the loop body never
    executes that int used to fall straight through to torch.isnan, raising
    TypeError. Reachable only once phiBar_fg stopped returning its sentinel on
    almost every call.
    """
    obs = OBS()
    # Choose a_j and Lambda so that ||p(0)|| == delta exactly, which makes
    # phi_bar(0) == 0 and satisfies the stopping test on the first evaluation.
    Lambda = torch.tensor([1.0, 2.0], dtype=DTYPE)
    a_j = torch.tensor([3.0, 8.0], dtype=DTYPE)
    delta = torch.norm(a_j / Lambda)

    x = obs.Newton(0, Lambda, a_j, delta)

    assert isinstance(x, torch.Tensor)
    assert torch.isfinite(x)
