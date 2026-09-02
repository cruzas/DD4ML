"""Smoke tests: run every standalone optimizer on problems with known solutions.

The point of this module is coverage of the *execution path*. Each optimizer is
driven for a fixed number of iterations on two textbook problems whose exact
minimisers are known in closed form:

    Rosenbrock   f(x, y) = (1 - x)^2 + 100 (y - x^2)^2      minimiser (1, 1), f* = 0
    SPD quadratic f(w)   = (w - w*)^T diag(1, 5) (w - w*)   minimiser (1, 2), f* = 0

Rosenbrock's curved, ill-conditioned valley is the standard stress case; the
quadratic is the easy control, so a method that fails on it is definitely broken
rather than merely under-iterated.

The four optimizers here minimise a function directly. The APTS family
(``apts_d``, ``apts_p``, ``apts_ip``, ``apts_pinn``) takes ``step(inputs, labels)``
against a model and criterion instead, so it needs a different harness and is
covered in ``test_apts_smoke.py``.

Thresholds below are empirical: each was measured before being asserted, and is
set well clear of the observed value so the tests do not flake.
"""

import pytest
import torch

from dd4ml.optimizers.asntr import ASNTR
from dd4ml.optimizers.lssr1_tr import LSSR1_TR
from dd4ml.optimizers.tr import TR
from dd4ml.optimizers.tradam import TRAdam

START = (-1.2, 1.0)  # the classic Rosenbrock starting point

ROSENBROCK_MIN = (1.0, 1.0)
QUADRATIC_MIN = (1.0, 2.0)


def rosenbrock(w):
    return (1.0 - w[0]) ** 2 + 100.0 * (w[1] - w[0] ** 2) ** 2


def quadratic(w):
    diff = w - torch.tensor(QUADRATIC_MIN, dtype=w.dtype)
    return (diff * diff * torch.tensor([1.0, 5.0], dtype=w.dtype)).sum()


PROBLEMS = {
    "rosenbrock": (rosenbrock, ROSENBROCK_MIN),
    "quadratic": (quadratic, QUADRATIC_MIN),
}


def _make_closures(w, fn):
    """Return (closure, zero_arg_closure) for the two calling conventions in use."""

    def closure(compute_grad=False):
        loss = fn(w)
        if compute_grad:
            if w.grad is not None:
                w.grad.zero_()
            loss.backward()
        return loss.detach()

    def zero_arg_closure():
        # TRAdam calls closure() with no arguments and reads p.grad afterwards.
        loss = fn(w)
        if w.grad is not None:
            w.grad.zero_()
        loss.backward()
        return loss.detach()

    return closure, zero_arg_closure


def _build(name, w, second_order):
    """Return (optimizer, step_callable) for one optimizer by name."""
    closure, zero_arg_closure = _make_closures(w, PROBLEMS[_build.problem][0])

    if name == "tr":
        opt = TR(
            [w],
            delta=0.5,
            max_delta=5.0,
            min_delta=1e-6,
            inc_factor=1.2,
            dec_factor=0.9,
            nu_dec=0.25,
            nu_inc=0.75,
            mem_length=5,
            norm_type=2,
            tol=1e-12,
            second_order=second_order,
        )
        return opt, lambda: opt.step(closure)

    if name == "lssr1_tr":
        opt = LSSR1_TR(
            [w],
            delta=0.5,
            min_delta=1e-6,
            max_delta=5.0,
            mem_length=5,
            max_wolfe_iters=5,
            max_zoom_iters=5,
            tol=1e-12,
            second_order=second_order,
        )
        return opt, lambda: opt.step(closure)

    if name == "tradam":
        opt = TRAdam([w], lr=0.05, norm_type=2)
        return opt, lambda: opt.step(zero_arg_closure)

    if name == "asntr":
        opt = ASNTR(
            [w],
            delta=0.5,
            min_delta=1e-6,
            max_delta=5.0,
            mem_length=5,
            tol=1e-12,
            second_order=second_order,
            # Shrink the non-monotonicity allowance so acceptance is governed by
            # actual decrease; see tests/test_asntr.py for the paper's Eqs. 6-10.
            c_1=1e-9,
            c_2=1e-9,
        )
        # With no subsampling, f_D == f_N and hNk == 0 selects the full-sample
        # branch of Algorithm 1, where acceptance depends on rho_N alone.
        return opt, lambda: opt.step(closure_main=closure, closure_d=closure, hNk=0.0)

    raise AssertionError(f"unknown optimizer {name!r}")


def _run(name, problem, iters, second_order=False):
    """Drive one optimizer and return (initial_loss, final_loss, final_iterate)."""
    fn, _ = PROBLEMS[problem]
    w = torch.nn.Parameter(torch.tensor(START, dtype=torch.float64))
    _build.problem = problem
    _, step = _build(name, w, second_order)

    initial = float(fn(w).detach())
    for _ in range(iters):
        step()
    final = float(fn(w).detach())
    return initial, final, w.detach().clone()


# --------------------------------------------------------------------------- #
# First-order paths: these all execute and make progress.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["tr", "lssr1_tr", "tradam", "asntr"])
@pytest.mark.parametrize("problem", ["quadratic", "rosenbrock"])
def test_optimizer_runs_and_reduces_objective(name, problem):
    """Every optimizer must run without error and lower the objective."""
    initial, final, w = _run(name, problem, iters=500)

    assert torch.isfinite(w).all(), f"{name} produced a non-finite iterate: {w}"
    assert final == final, f"{name} produced NaN loss"  # NaN check
    assert final < initial, (
        f"{name} on {problem}: objective did not decrease ({initial} -> {final})"
    )


@pytest.mark.parametrize(
    ("name", "tol"),
    [
        # Measured over 500 iterations from the standard start; each bound is
        # roughly an order of magnitude above what the method actually reaches.
        ("tr", 1e-6),
        ("tradam", 1e-6),
        ("asntr", 1e-6),
        # LSSR1_TR is an L-SR1 method: with second_order disabled it has no
        # curvature model to work with and plateaus around f ~ 0.9 instead of
        # converging. Its intended second-order path is covered (as an expected
        # failure) below. Only progress is asserted here.
        pytest.param("lssr1_tr", None, id="lssr1_tr-progress-only"),
    ],
)
def test_converges_on_quadratic(name, tol):
    """The SPD quadratic has a unique minimiser at (1, 2)."""
    initial, final, w = _run(name, "quadratic", iters=500)

    if tol is None:
        assert final < initial / 2.0, f"{name} made too little progress"
        return

    assert final < tol, f"{name} did not reach the minimum: f = {final}"
    expected = torch.tensor(QUADRATIC_MIN, dtype=w.dtype)
    assert torch.allclose(w, expected, atol=1e-3), f"{name} converged to {w}"


@pytest.mark.parametrize("name", ["tr", "tradam", "asntr"])
def test_makes_substantial_progress_on_rosenbrock(name):
    """Rosenbrock is hard; require a large reduction rather than convergence."""
    initial, final, w = _run(name, "rosenbrock", iters=500)

    assert final < initial / 10.0, (
        f"{name} on Rosenbrock: {initial} -> {final} is too little progress"
    )


def test_tradam_solves_rosenbrock_given_enough_iterations():
    """With a longer budget TRAdam reaches the Rosenbrock minimiser (1, 1)."""
    _, final, w = _run("tradam", "rosenbrock", iters=2000)

    assert final < 1e-4, f"expected convergence, got f = {final}"
    expected = torch.tensor(ROSENBROCK_MIN, dtype=w.dtype)
    assert torch.allclose(w, expected, atol=1e-2), f"converged to {w}"


# --------------------------------------------------------------------------- #
# Second-order paths: currently broken, pinned as strict xfails.
# --------------------------------------------------------------------------- #


# The second-order path is broken for every optimizer that uses it. The root
# cause is in the LSR1/OBS pair: OBS.solve_tr_subproblem takes a Cholesky factor
# of Psi^T Psi, but LSR1 never rejects curvature pairs whose psi is linearly
# dependent on the existing basis -- that check is commented out in lsr1.py, as
# is the Tikhonov regularisation in obs.py. Once the iterates stop varying much,
# successive pairs become near-parallel and Psi loses rank. This is
# dimension-independent: it reproduces at n = 50 with a memory of 10 just as it
# does at n = 2, and disappears only when the memory holds a single pair (pinned
# by test_second_order_works_with_a_single_memory_pair below).
#
# The *symptom* is platform-dependent, so these markers deliberately do not pin
# an exception type. The same case surfaces as any of:
#   * torch.linalg.LinAlgError  -- the Cholesky rejects the rank-deficient input;
#   * ValueError                -- OBS.Newton returns a non-finite root, which
#                                  propagates into the next gradient;
#   * plain divergence          -- no exception at all, the objective runs away
#                                  (asntr on Rosenbrock reached ~1.9e10 on
#                                  Linux/x86 while raising LinAlgError on
#                                  macOS/ARM).
# Which one you get depends on the BLAS and the platform, so pinning `raises=`
# made this suite pass locally and fail in CI. What is stable is that the path
# does not work, and that is what these markers assert.
_SECOND_ORDER_BROKEN = (
    "Second-order path is broken; see the comment above this marker. "
    "Remove the xfail once LSR1/OBS is repaired."
)

SECOND_ORDER_BROKEN = pytest.mark.xfail(strict=True, reason=_SECOND_ORDER_BROKEN)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        pytest.param("tr", "quadratic", marks=SECOND_ORDER_BROKEN),
        pytest.param("lssr1_tr", "quadratic", marks=SECOND_ORDER_BROKEN),
        pytest.param("asntr", "quadratic", marks=SECOND_ORDER_BROKEN),
        pytest.param("tr", "rosenbrock", marks=SECOND_ORDER_BROKEN),
        pytest.param("lssr1_tr", "rosenbrock", marks=SECOND_ORDER_BROKEN),
        pytest.param("asntr", "rosenbrock", marks=SECOND_ORDER_BROKEN),
    ],
)
def test_second_order_paths(name, problem):
    """Second-order mode is the default for LSSR1_TR and ASNTR."""
    initial, final, w = _run(name, problem, iters=500, second_order=True)

    assert torch.isfinite(w).all(), f"{name} produced a non-finite iterate: {w}"
    assert final < initial, f"{name} on {problem} (second order): {initial} -> {final}"


@pytest.mark.parametrize("name", ["tr", "lssr1_tr", "asntr"])
def test_second_order_works_with_a_single_memory_pair(name):
    """A memory of one pair keeps Psi full rank, which isolates the cause above."""
    fn, _ = PROBLEMS["quadratic"]
    w = torch.nn.Parameter(torch.tensor(START, dtype=torch.float64))
    closure, _unused = _make_closures(w, fn)

    common = {
        "delta": 0.5,
        "min_delta": 1e-6,
        "max_delta": 5.0,
        "mem_length": 1,
        "second_order": True,
        "tol": 1e-12,
    }
    if name == "tr":
        opt = TR(
            [w],
            inc_factor=1.2,
            dec_factor=0.9,
            nu_dec=0.25,
            nu_inc=0.75,
            norm_type=2,
            **common,
        )
        step = lambda: opt.step(closure)  # noqa: E731
    elif name == "lssr1_tr":
        opt = LSSR1_TR([w], max_wolfe_iters=5, max_zoom_iters=5, **common)
        step = lambda: opt.step(closure)  # noqa: E731
    else:
        opt = ASNTR([w], c_1=1e-9, c_2=1e-9, **common)
        step = lambda: opt.step(  # noqa: E731
            closure_main=closure, closure_d=closure, hNk=0.0
        )

    initial = float(fn(w).detach())
    for _ in range(200):
        step()
    final = float(fn(w).detach())

    assert torch.isfinite(w).all()
    assert final < initial, (
        f"{name} (mem_length=1) did not decrease: {initial} -> {final}"
    )
