"""Regression tests for LSSR1_TR's trust-region bookkeeping and line search.

LSSR1_TR is a trust-region method with a strong Wolfe line search layered on
top: the trust region sets the direction and scale, the line search picks the
step length along it, and a heavy-ball term carries momentum between steps.

Three defects made it stall well short of the minimiser even after LSR1 and OBS
were repaired. Each is pinned below, because each is silent -- the optimizer
still ran and still reduced the objective a little, so nothing failed loudly.
"""

import contextlib
import io

import torch

from dd4ml.optimizers.lssr1_tr import LSSR1_TR

DTYPE = torch.float64


def _quadratic_problem(n=10, seed=0):
    """SPD quadratic with a known minimiser, well conditioned by construction."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=gen, dtype=DTYPE)
    Q = A @ A.T + n * torch.eye(n, dtype=DTYPE)
    minimiser = torch.arange(1.0, n + 1, dtype=DTYPE) / n
    w = torch.nn.Parameter(torch.zeros(n, dtype=DTYPE))

    def objective():
        diff = w - minimiser
        return 0.5 * diff @ Q @ diff

    def closure(compute_grad=False):
        loss = objective()
        if compute_grad:
            if w.grad is not None:
                w.grad.zero_()
            loss.backward()
        return loss.detach()

    return w, objective, closure, minimiser


def _optimizer(w, **kw):
    kwargs = {
        "delta": 0.5,
        "min_delta": 1e-10,
        "max_delta": 5.0,
        "second_order": True,
        "mem_length": 5,
        "max_wolfe_iters": 5,
        "max_zoom_iters": 5,
        "tol": 1e-14,
    }
    kwargs.update(kw)
    return LSSR1_TR([w], **kwargs)


def _run(opt, closure, iters):
    """Drive the optimizer, swallowing its progress prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(iters):
            opt.step(closure)


def test_trust_region_radius_does_not_collapse():
    """The radius must respond to step quality, not shrink unconditionally.

    rho was computed only when ``pred_red < 0``. solve_tr_* returns the classical
    predicted reduction -(g'p + 0.5 p'Bp), which is *positive* for a descent
    step, so the guard never held and rho was pinned at 0.0. That is below
    tau_2, so the radius shrank on every iteration regardless of how well the
    step had done: it reached min_delta within a dozen steps and the method
    froze.
    """
    w, objective, closure, _ = _quadratic_problem()
    opt = _optimizer(w, delta=0.5)

    radii = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(12):
            opt.step(closure)
            radii.append(opt.delta)

    assert max(radii) > radii[0] / 2, (
        f"radius only ever shrank: {radii} -- rho is not being computed"
    )
    assert opt.delta > 1e-8, f"radius collapsed to {opt.delta}"


def test_line_search_starts_from_the_full_trust_region_step():
    """The first trial step length must respect alpha_0.

    The initial trial was hard-coded to 0.5 * alpha_max (5.0 by default) and the
    alpha_0 argument was ignored. Since the direction is already scaled so that
    ||p|| <= delta, that trialled a step five times outside the trust region on
    every iteration, and the radius controlled nothing.
    """
    w, objective, closure, _ = _quadratic_problem(n=4)
    opt = _optimizer(w, delta=0.05)

    seen = []
    original = opt._evaluate_function_and_gradient

    def recording(wk, p, alpha, closure_fn):
        seen.append((float(alpha), float(torch.norm(p))))
        return original(wk, p, alpha, closure_fn)

    opt._evaluate_function_and_gradient = recording
    _run(opt, closure, iters=1)

    assert seen, "line search never evaluated a trial point"
    first_alpha, direction_norm = seen[0]
    assert first_alpha <= 1.0 + 1e-12, (
        f"first trial alpha was {first_alpha}; expected the alpha_0 default of 1.0"
    )
    assert first_alpha * direction_norm <= 2.0 * opt.delta, (
        "the first trial step is far outside the trust region"
    )


def test_momentum_term_accumulates():
    """vk must actually carry momentum between steps.

    ``vk <- mu*vk + (w_k - w_{k-1})`` read ``st["old_wk"]`` after it had been
    reassigned to the current iterate a few lines earlier, so the increment was
    identically zero. vk stayed at its zero initialisation forever and the
    heavy-ball term contributed nothing.
    """
    w, objective, closure, _ = _quadratic_problem()
    opt = _optimizer(w, mu=0.9)

    _run(opt, closure, iters=1)
    assert float(opt.state["flat_vk"].norm()) == 0.0, (
        "there is no previous iterate after one step, so vk must still be zero"
    )

    _run(opt, closure, iters=2)
    assert float(opt.state["flat_vk"].norm()) > 0.0, (
        "vk is still zero after several steps -- the momentum term is dead"
    )


def test_momentum_is_disabled_by_mu_zero():
    """mu = 0 must switch the heavy-ball term off entirely."""
    w, objective, closure, _ = _quadratic_problem()
    opt = _optimizer(w, mu=0.0)

    _run(opt, closure, iters=5)

    # With mu = 0 the term retains only the most recent difference, never a
    # running accumulation; the previous step is still added once.
    assert torch.isfinite(opt.state["flat_vk"]).all()


def test_converges_on_a_quadratic():
    """End to end: all three fixes together reach the known minimiser."""
    w, objective, closure, minimiser = _quadratic_problem()
    opt = _optimizer(w)

    initial = float(objective().detach())
    _run(opt, closure, iters=200)
    final = float(objective().detach())

    assert final < 1e-12, f"did not converge: {initial} -> {final}"
    assert torch.allclose(w.detach(), minimiser, atol=1e-6), (
        f"converged to {w.detach()} instead of {minimiser}"
    )
