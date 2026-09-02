"""Unit tests for the ASNTR optimizer.

The reference is:

    N. Krejic, N. Krklec Jerinkic, A. Martinez, M. Yousefi,
    "A non-monotone trust-region method with noisy oracles and additional
    sampling", Computational Optimization and Applications 89:247-278 (2024).
    https://doi.org/10.1007/s10589-024-00580-w

Equation numbers below refer to that paper:

    (4)  Q_k(p) = 0.5 p^T B_k p + g_k^T p,      minimised over ||p|| <= delta_k
    (6)  rho_{N_k} = (f_{N_k}(w_t) - r_{N_k}) / Q_k(p_k)
    (7)  r_{N_k}   = f_{N_k}(w_k) + t_k * delta_k
    (9)  rho_{D_k} = (f_{D_k}(w_t) - r_{D_k}) / L_k(-g_bar_k),  L_k(v) = v^T g_bar_k
    (10) r_{D_k}   = f_{D_k}(w_k) + delta_k * ttilde_k

and Algorithm 1 lines 23-34 (acceptance) and 36-42 (trust-region update).

These tests pin the orientation of the two ratios. An earlier implementation had
f(w_k) and f(w_t) transposed in both (6)/(7) and (9)/(10), which flips the sign
of each ratio and therefore rejects exactly the steps it should accept.
"""

import pytest
import torch

from dd4ml.optimizers.asntr import ASNTR

# The flat buffers adopt the parameter dtype, so float64 arithmetic stays
# float64 end to end. This tolerance only absorbs ordinary floating-point
# rounding; it is far tighter than the ~1e-8 error a float32 buffer would
# introduce, so a regression in the buffer dtype fails these tests.
ATOL = 1e-12

# Make t_k and ttilde_k negligible so the non-monotonicity allowance does not
# mask the orientation of the ratios; the paper only requires them to be
# positive and summable (Eqs. 8 and 11).
TINY_NONMONOTONE = {"c_1": 1e-9, "c_2": 1e-9, "alpha": 1.1}


def _make_problem(w0, n_scale=1.0, d_center=0.0, d_scale=1.0):
    """Return (param, closure_main, closure_d) for two quadratics.

    f_N(w) = 0.5 * n_scale * ||w||^2               (minimiser at 0)
    f_D(w) = 0.5 * d_scale * ||w - d_center||^2    (minimiser at d_center)

    Each closure writes its own gradient into ``param.grad`` when asked, which
    is what ASNTR's flat-gradient hook reads back.
    """
    param = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float64))

    def closure_main(compute_grad=False):
        loss = 0.5 * n_scale * (param * param).sum()
        if compute_grad:
            if param.grad is not None:
                param.grad.zero_()
            loss.backward()
        return loss.detach()

    def closure_d(compute_grad=False):
        diff = param - d_center
        loss = 0.5 * d_scale * (diff * diff).sum()
        if compute_grad:
            if param.grad is not None:
                param.grad.zero_()
            loss.backward()
        return loss.detach()

    return param, closure_main, closure_d


def _opt(param, delta=0.1, **kw):
    kwargs = {
        "device": "cpu",
        "delta": delta,
        "min_delta": 1e-6,
        "max_delta": 10.0,
        "second_order": False,
        "norm_type": 2,
        **TINY_NONMONOTONE,
    }
    kwargs.update(kw)
    return ASNTR([param], **kwargs)


def test_improving_step_is_accepted_full_sample():
    """Eq. (6)/(7): a step that lowers f_N must be accepted.

    Hand computation at w=1, delta=0.1, f_N(w)=0.5 w^2:
        g = 1, ||g|| = 1, p = -delta*g/||g|| = -0.1, w_t = 0.9
        Q_k(p) = g^T p = -0.1                        (Eq. 4, negative)
        f_N(w_k) = 0.5, f_N(w_t) = 0.405
        r_{N_k} ~= 0.5                               (Eq. 7, t_k ~ 0)
        rho_N = (0.405 - 0.5) / (-0.1) = +0.95       (Eq. 6)
    which is >= eta, so the step is accepted. With f(w_k)/f(w_t) transposed the
    ratio is -0.95 and the step would be rejected.
    """
    param, cm, cd = _make_problem([1.0])
    opt = _opt(param, delta=0.1)

    # hNk == 0 selects the full-sample branch (Algorithm 1 lines 29-34), where
    # acceptance depends on rho_N alone.
    opt.step(closure_main=cm, closure_d=cd, hNk=0.0)

    assert param.detach().item() == pytest.approx(0.9, abs=ATOL), (
        "improving trial point should be kept"
    )


def test_trust_region_grows_on_very_good_step():
    """Algorithm 1 lines 38-39: rho_N > eta_2 and ||p|| >= tau_2*delta enlarges delta."""
    param, cm, cd = _make_problem([1.0])
    opt = _opt(param, delta=0.1, eta_2=0.75, tau_2=0.8, tau_3=2.0)

    opt.step(closure_main=cm, closure_d=cd, hNk=0.0)

    # rho_N = 0.95 > eta_2, and ||p|| = 0.1 > tau_2*delta = 0.08.
    assert opt.delta == pytest.approx(0.2)


def test_worsening_step_is_rejected_and_radius_shrinks():
    """A trial point that raises f_N gives rho_N < eta_1: reject and shrink.

    Starting at w = -0.05 with delta = 0.2, the step overshoots the minimiser to
    w_t = 0.15, so f_N increases and the iterate must be restored.
    """
    param, cm, cd = _make_problem([-0.05])
    opt = _opt(param, delta=0.2, eta_1=0.1, tau_1=0.5)

    opt.step(closure_main=cm, closure_d=cd, hNk=0.0)

    assert param.detach().item() == pytest.approx(-0.05, abs=ATOL), (
        "rejected step must restore w_k"
    )
    assert opt.delta == pytest.approx(0.1), (
        "rho_N < eta_1 must shrink the radius by tau_1"
    )


def test_rho_d_can_veto_an_otherwise_good_step():
    """Eq. (9)/(10): when subsampled, both ratios must clear their thresholds.

    f_D is centred at 2.0, so stepping 1.0 -> 0.9 moves away from the D-minimiser:
        f_D(w_k) = 0.5, f_D(w_t) = 0.605, g_bar = -1
        L_k(-g_bar) = -||g_bar||^2 = -1              (Eq. 9 denominator)
        r_{D_k} ~= 0.5                               (Eq. 10, ttilde_k ~ 0)
        rho_D = (0.605 - 0.5) / (-1) = -0.105 < nu
    so the step is rejected even though rho_N = +0.95 clears eta.
    """
    param, cm, cd = _make_problem([1.0], d_center=2.0)
    opt = _opt(param, delta=0.1, nu=1e-4)

    # hNk != 0 selects the subsampled branch (Algorithm 1 lines 23-28).
    opt.step(closure_main=cm, closure_d=cd, hNk=0.5)

    assert param.detach().item() == pytest.approx(1.0, abs=ATOL), (
        "rho_D < nu must veto the step"
    )


def test_rho_d_below_nu_requests_a_larger_sample():
    """Algorithm 1 lines 10-11: rho_D < nu increases the sample size."""
    param, cm, cd = _make_problem([1.0], d_center=2.0)
    opt = _opt(param, delta=0.1, nu=1e-4)

    opt.step(closure_main=cm, closure_d=cd, hNk=0.5)

    assert opt.inc_batch_size is True
    assert opt.move_to_next_batch is True


def test_agreeing_subsample_accepts_the_step():
    """With f_D agreeing with f_N, the subsampled branch accepts as well."""
    param, cm, cd = _make_problem([1.0], d_center=0.0)
    opt = _opt(param, delta=0.1)

    opt.step(closure_main=cm, closure_d=cd, hNk=0.5)

    assert param.detach().item() == pytest.approx(0.9, abs=ATOL)
    assert opt.inc_batch_size is False


def test_converges_on_quadratic():
    """Repeated steps drive the iterate toward the minimiser of f_N."""
    param, cm, cd = _make_problem([1.0])
    opt = _opt(param, delta=0.25)

    start = abs(param.detach().item())
    for _ in range(40):
        opt.step(closure_main=cm, closure_d=cd, hNk=0.0)

    end = abs(param.detach().item())
    assert end < start / 10.0, f"expected decrease toward 0, got {start} -> {end}"


def test_zero_gradient_is_handled():
    """At a stationary point the solver returns a zero step and nothing blows up."""
    param, cm, cd = _make_problem([0.0])
    opt = _opt(param, delta=0.1)

    opt.step(closure_main=cm, closure_d=cd, hNk=0.0)

    assert param.detach().item() == pytest.approx(0.0, abs=ATOL)
    assert torch.isfinite(torch.tensor(opt.delta))


def test_flat_buffers_adopt_the_parameter_dtype():
    """Regression: the buffers used to be allocated with the global default.

    ASNTR stages every parameter and gradient through state["flat_wk"] and
    state["flat_gk"]. Those were allocated with torch.zeros(..., device=...) and
    no dtype, so they came out float32 regardless of the model. A float64 model
    was therefore truncated on every flatten/unflatten round trip: restoring the
    iterate after a rejected step returned -0.05 as -0.05000000074505806.
    """
    param, cm, cd = _make_problem([-0.05])
    opt = _opt(param, delta=0.2)

    assert opt.state["flat_wk"].dtype == torch.float64
    assert opt.state["flat_gk"].dtype == torch.float64

    # This step is rejected (it overshoots the minimiser), so w_k must come back
    # bit-for-bit, not merely to within float32 precision.
    opt.step(closure_main=cm, closure_d=cd, hNk=0.0)
    assert param.detach().item() == -0.05


def test_flat_buffers_follow_float32_parameters():
    """A float32 model must not be silently widened either."""
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    opt = _opt(param, delta=0.1)

    assert opt.state["flat_wk"].dtype == torch.float32


def test_sr1_memory_keeps_the_parameter_dtype():
    """The L-SR1 memory must not downcast a float64 model.

    LSR1 defaulted self.dtype to float32 and update_memory() casts every
    incoming curvature pair to it, so ASNTR's Hessian approximation ran in
    single precision even once its own flat buffers were fixed.

    The problem must be anisotropic. For an isotropic f = 0.5||w||^2 the true
    Hessian is a multiple of the identity, so psi = y - gamma*s is exactly zero
    and LSR1 rightly rejects every pair as carrying no information beyond gamma.
    second_order stays off: update_memory() runs either way, and leaving it on
    would hit the known OBS rank-deficiency break.
    """
    w = torch.nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64))
    curvature = torch.tensor([1.0, 5.0], dtype=torch.float64)

    def closure(compute_grad=False):
        loss = (w * w * curvature).sum()
        if compute_grad:
            if w.grad is not None:
                w.grad.zero_()
            loss.backward()
        return loss.detach()

    opt = ASNTR(
        [w],
        device="cpu",
        delta=0.2,
        min_delta=1e-6,
        max_delta=5.0,
        second_order=False,
        mem_length=4,
        tol=1e-12,
        **TINY_NONMONOTONE,
    )

    assert opt.hess.dtype == torch.float64
    assert opt.hess.gamma.dtype == torch.float64

    for _ in range(5):
        opt.step(closure_main=closure, closure_d=closure, hNk=0.0)

    assert opt.hess._S, "expected at least one stored curvature pair"
    assert all(v.dtype == torch.float64 for v in opt.hess._S)
    assert all(v.dtype == torch.float64 for v in opt.hess._Y)
    assert opt.hess._S_matrix.dtype == torch.float64
    assert opt.hess._Y_matrix.dtype == torch.float64
    assert opt.hess.gamma.dtype == torch.float64


def test_lsr1_adopts_dtype_from_data_when_none_is_given():
    """A caller that forgets to pass dtype still gets the right one."""
    from dd4ml.optimizers.lsr1 import LSR1

    hess = LSR1(gamma=1.0, memory_length=3, tol=1e-12)
    s = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.5, 0.25], dtype=torch.float64)

    hess.update_memory(s, y)

    assert hess.dtype == torch.float64
    assert hess.gamma.dtype == torch.float64
    assert hess._S_matrix.dtype == torch.float64


def test_lsr1_honours_an_explicit_dtype():
    """An explicit dtype wins over the incoming data."""
    from dd4ml.optimizers.lsr1 import LSR1

    hess = LSR1(gamma=1.0, memory_length=3, tol=1e-12, dtype=torch.float32)
    s = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.5, 0.25], dtype=torch.float64)

    hess.update_memory(s, y)

    assert hess.dtype == torch.float32
    assert hess._S_matrix.dtype == torch.float32
