import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset
from dd4ml.models.ffnn.pinn_ffnn import PINNFFNN
from dd4ml.optimizers.apts_pinn import APTS_PINN
from dd4ml.optimizers.tr import TR
from dd4ml.utility.pinn_allencahn_loss import AllenCahnPINNLoss
from dd4ml.utility.dist_utils import prepare_distributed_environment


def test_split_domain():
    cfg = AllenCahn1DDataset.get_default_config()
    ds = AllenCahn1DDataset(cfg)
    subs = ds.split_domain(2)
    assert len(subs) == 2
    # first subdomain should start at global low, last at global high
    assert torch.isclose(subs[0].x_boundary[0], torch.tensor([cfg.low]))
    assert torch.isclose(subs[-1].x_boundary[-1], torch.tensor([cfg.high]))


def _run_apts_pinn(rank: int, world_size: int):
    prepare_distributed_environment(
        rank=rank,
        master_addr="localhost",
        master_port="12355",
        world_size=world_size,
        is_cuda_enabled=False,
    )

    cfg = AllenCahn1DDataset.get_default_config()
    ds = AllenCahn1DDataset(cfg)
    model_cfg = PINNFFNN.get_default_config()
    model = PINNFFNN(model_cfg)
    criterion = AllenCahnPINNLoss()

    tr_kwargs = dict(
        delta=0.1,
        nu_dec=0.25,
        nu_inc=0.75,
        inc_factor=1.2,
        dec_factor=0.9,
        max_delta=2.0,
        min_delta=1e-3,
        tol=1e-6,
    )

    opt = APTS_PINN(
        model.parameters(),
        model=model,
        criterion=criterion,
        device="cpu",
        glob_opt=TR,
        glob_opt_hparams=tr_kwargs,
        loc_opt=TR,
        loc_opt_hparams=tr_kwargs,
        glob_pass=False,
        foc=False,
        norm_type=2,
        max_loc_iters=1,
        max_glob_iters=1,
        num_subdomains=world_size,
        **tr_kwargs,
    )

    x = ds.data
    boundary_flag = torch.cat(
        [
            torch.zeros(len(ds.x_interior), 1),
            torch.ones(len(ds.x_boundary), 1),
        ]
    )
    criterion.current_x = x
    loss = opt.step(inputs=x, labels=boundary_flag)
    assert torch.isfinite(loss)
    dist.destroy_process_group()


def test_apts_pinn_step_runs():
    world_size = 2
    mp.spawn(_run_apts_pinn, args=(world_size,), nprocs=world_size, join=True)
