"""Smoke tests for the APTS optimizer family.

Unlike the trust-region optimizers in ``test_optimizers_smoke.py``, the APTS
methods do not minimise a bare function: they take ``step(inputs, labels)`` and
drive a model through a criterion. The problem with a known solution used here
is therefore a supervised one -- two tight, well-separated Gaussian clusters,
whose exact solution is "every point classified correctly, loss 0".

These tests initialise a real single-rank ``gloo`` process group rather than
mocking ``torch.distributed``, so the collective calls inside APTS execute for
real (on one rank a reduction is a no-op, but the code path is taken).
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from dd4ml.models.ffnn.simple_ffnn import SimpleFFNN
from dd4ml.optimizers.apts_d import APTS_D
from dd4ml.optimizers.apts_p import APTS_P
from dd4ml.optimizers.tr import TR
from dd4ml.utility.optimizer_utils import get_loc_tr_hparams, get_tr_hparams

CLUSTER_SIZE = 32
SEED = 0


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    """A real single-rank gloo group.

    File-based rendezvous avoids picking a TCP port, which would be flaky on a
    shared CI runner.
    """
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("dist") / "rendezvous"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


class _HParams:
    """Minimal stand-in for the config object the hparam helpers read."""

    delta = 0.1
    max_delta = 2.0
    min_delta = 1e-4
    norm_type = 2
    tol = 1e-10
    glob_second_order = False
    glob_dogleg = False
    loc_second_order = False
    loc_dogleg = False


def _make_separable_problem():
    """Two tight clusters at (-2, -2) and (2, 2); exact solution is 100% accuracy."""
    torch.manual_seed(SEED)
    lo = torch.randn(CLUSTER_SIZE, 2) * 0.15 + torch.tensor([-2.0, -2.0])
    hi = torch.randn(CLUSTER_SIZE, 2) * 0.15 + torch.tensor([2.0, 2.0])
    inputs = torch.cat([lo, hi])
    labels = torch.cat(
        [
            torch.zeros(CLUSTER_SIZE, dtype=torch.long),
            torch.ones(CLUSTER_SIZE, dtype=torch.long),
        ]
    )
    return inputs, labels


def _make_model():
    cfg = SimpleFFNN.get_default_config()
    cfg.input_features = 2
    cfg.output_classes = 2
    cfg.fc_layers = [8]
    cfg.model_type = None
    torch.manual_seed(SEED)
    return SimpleFFNN(cfg)


def _accuracy(model, inputs, labels):
    with torch.no_grad():
        return (model(inputs).argmax(dim=1) == labels).float().mean().item()


@pytest.mark.distributed
@pytest.mark.parametrize("cls", [APTS_D, APTS_P], ids=["apts_d", "apts_p"])
def test_apts_solves_a_separable_problem(process_group, cls):
    """APTS must run end to end and reach the known solution."""
    inputs, labels = _make_separable_problem()
    model = _make_model()
    # SimpleFFNN.forward ends in log_softmax, so NLL is the matching criterion.
    criterion = nn.NLLLoss()

    kwargs = {
        "params": model.parameters(),
        "model": model,
        "criterion": criterion,
        "device": "cpu",
        "nr_models": 1,
        "glob_opt": TR,
        "glob_opt_hparams": get_tr_hparams(_HParams),
        "loc_opt": TR,
        "loc_opt_hparams": get_loc_tr_hparams(_HParams),
        "glob_pass": True,
        "norm_type": 2,
        "max_loc_iters": 3,
        "max_glob_iters": 1,
        "tol": 1e-10,
        "delta": 0.1,
        "min_delta": 1e-4,
        "max_delta": 2.0,
    }
    if cls is APTS_D:
        # First-order consistency correction; APTS_P does not take this flag.
        kwargs["foc"] = True

    optimizer = cls(**kwargs)

    with torch.no_grad():
        initial_loss = float(criterion(model(inputs), labels))

    for _ in range(60):
        optimizer.step(inputs, labels)

    with torch.no_grad():
        final_loss = float(criterion(model(inputs), labels))

    assert final_loss == final_loss, "loss became NaN"
    assert final_loss < initial_loss, f"{initial_loss} -> {final_loss}"
    assert _accuracy(model, inputs, labels) == 1.0, (
        "the clusters are linearly separable; the exact solution classifies all "
        "points correctly"
    )
    assert all(torch.isfinite(p).all() for p in model.parameters()), (
        "model parameters went non-finite"
    )


@pytest.mark.distributed
def test_apts_d_leaves_parameters_finite_with_glob_pass_disabled(process_group):
    """The no-global-pass branch is a separate code path; check it also runs."""
    inputs, labels = _make_separable_problem()
    model = _make_model()
    criterion = nn.NLLLoss()

    optimizer = APTS_D(
        params=model.parameters(),
        model=model,
        criterion=criterion,
        device="cpu",
        nr_models=1,
        glob_opt=TR,
        glob_opt_hparams=get_tr_hparams(_HParams),
        loc_opt=TR,
        loc_opt_hparams=get_loc_tr_hparams(_HParams),
        glob_pass=False,
        foc=False,
        norm_type=2,
        max_loc_iters=3,
        max_glob_iters=1,
        tol=1e-10,
        delta=0.1,
        min_delta=1e-4,
        max_delta=2.0,
    )

    with torch.no_grad():
        initial_loss = float(criterion(model(inputs), labels))

    for _ in range(60):
        optimizer.step(inputs, labels)

    with torch.no_grad():
        final_loss = float(criterion(model(inputs), labels))

    assert final_loss < initial_loss, f"{initial_loss} -> {final_loss}"
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.skip(
    reason=(
        "APTS_IP drives a PMW ParallelizedModel: it calls subdomain_params(), "
        "subdomain_forward(), subdomain_backward(), sync_params() and reads "
        "model.model_handler. Covering it needs the model-parallel harness and "
        "more than one rank, not the single-rank group used here."
    )
)
def test_apts_ip_smoke():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "APTS_PINN needs a PINN criterion carrying domain bounds (low/high) and "
        "the collocation-point batching convention, so its known-solution case "
        "is a PDE residual rather than the classification problem used here. "
        "See tests/test_pinn_subdomains.py for the PINN-side coverage."
    )
)
def test_apts_pinn_smoke():
    raise AssertionError("unreachable")
