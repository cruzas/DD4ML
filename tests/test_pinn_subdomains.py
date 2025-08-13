import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset
from dd4ml.utility.dist_utils import prepare_distributed_environment
from dd4ml.utility.pinn_allencahn_loss import AllenCahnPINNLoss


def test_split_domain():
    cfg = AllenCahn1DDataset.get_default_config()
    ds = AllenCahn1DDataset(cfg)
    subs = ds.split_domain(2)
    assert len(subs) == 2
    # first subdomain should start at global low, last at global high
    assert torch.isclose(subs[0].x_boundary[0], torch.tensor([cfg.low]))
    assert torch.isclose(subs[-1].x_boundary[-1], torch.tensor([cfg.high]))
    for sub in subs:
        # Each subdomain should contain n_interior + 2 total points
        assert len(sub) == sub.config.n_interior + sub.config.n_boundary
        all_points = torch.cat([sub.x_interior, sub.x_boundary])
        # Interior points should exclude boundaries (no duplicates)
        assert torch.unique(all_points).numel() == all_points.numel()


def test_split_domain_exclusive():
    cfg = AllenCahn1DDataset.get_default_config()
    ds = AllenCahn1DDataset(cfg)
    subs = ds.split_domain(2, exclusive=True)
    # Ensure the combined subdomains cover the domain without overlap
    all_points = torch.cat([sub.data for sub in subs])
    assert torch.unique(all_points).numel() == all_points.numel()


def test_split_domain_order():
    cfg = AllenCahn1DDataset.get_default_config()
    ds = AllenCahn1DDataset(cfg)
    subs = ds.split_domain(2, exclusive=True)
    for sub in subs:
        xs = sub.data[:, 0]
        assert torch.all(torch.diff(xs) >= 0), "Subdomain points are not ordered"


def test_apts_pinn_dataset_not_split():
    """Ensure dataset is not split when using the APTS_PINN optimizer."""
    from unittest.mock import patch

    from dd4ml.utility.trainer_setup import get_config_model_and_trainer
    from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset

    args = {
        "dataset_name": "allencahn1d",
        "model_name": "pinn_ffnn",
        "optimizer": "apts_pinn",
        "criterion": "pinn_allencahn",
        "batch_size": 12,
        "effective_batch_size": 12,
        "epochs": 1,
        "max_iters": 1,
        "learning_rate": 0.1,
        "seed": 42,
        "num_subdomains": 2,
        "num_stages": 1,
        "num_replicas_per_subdomain": 1,
        "gradient_accumulation": False,
        "accumulation_steps": 1,
        "batch_inc_factor": 1.0,
        "overlap": 0.0,
        "contiguous_subdomains": True,
        "exclusive": True,
        "glob_opt": "lssr1_tr",
        "loc_opt": "lssr1_tr",
    }

    with patch("torch.distributed.is_initialized", return_value=False), patch(
        "dd4ml.utility.trainer_setup.get_device", return_value="cpu"
    ), patch("dd4ml.trainer.dist.get_backend", return_value="gloo"), patch(
        "dd4ml.datasets.pinn_allencahn.AllenCahn1DDataset.split_domain",
        wraps=AllenCahn1DDataset.split_domain,
    ) as mock_split:
        _, _, trainer = get_config_model_and_trainer(args, None)
        mock_split.assert_not_called()
        assert len(trainer.train_dataset) == 12


def _run_apts_pinn(rank: int, world_size: int, epochs: int):
    from dd4ml.models.ffnn.pinn_ffnn import PINNFFNN
    from dd4ml.optimizers.apts_pinn import APTS_PINN
    from dd4ml.optimizers.tr import TR

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
        max_loc_iters=10,
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

    for epoch in range(epochs):
        loss = opt.step(inputs=x, labels=boundary_flag)
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        assert torch.isfinite(loss), f"Loss is not finite at epoch {epoch}"

    dist.destroy_process_group()


def test_run_by_epoch_pinn_full_dataset_overlap():
    """run_by_epoch_PINN should process the full local dataset when the
    sampler does not shard data (e.g. domain decomposition with overlap)."""
    from unittest.mock import patch
    from torch.utils.data.distributed import DistributedSampler
    from dd4ml.utility.trainer_setup import get_config_model_and_trainer

    args = {
        "dataset_name": "allencahn1d",
        "model_name": "pinn_ffnn",
        "optimizer": "tr",
        "criterion": "pinn_allencahn",
        "batch_size": 12,
        "effective_batch_size": 12,
        "epochs": 0,
        "max_iters": 1,
        "learning_rate": 0.1,
        "seed": 42,
        "num_subdomains": 2,
        "num_stages": 1,
        "num_replicas_per_subdomain": 1,
        "gradient_accumulation": False,
        "accumulation_steps": 1,
        "batch_inc_factor": 1.0,
        "overlap": 0.0,
        "contiguous_subdomains": True,
        "exclusive": False,
        "delta": 0.1,
        "max_delta": 2.0,
        "min_delta": 1e-3,
        "tol": 1e-6,
    }

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ), patch("torch.distributed.get_rank", return_value=0), patch(
        "dd4ml.utility.trainer_setup.get_device", return_value="cpu"
    ), patch("dd4ml.trainer.dist.get_backend", return_value="gloo"), patch(
        "dd4ml.utility.trainer_setup.DDP", lambda m, *a, **k: m
    ):
        _, _, trainer = get_config_model_and_trainer(args, None)
        trainer.setup_data_loaders()
        assert not isinstance(trainer.train_loader.sampler, DistributedSampler)
        with patch.object(
            trainer,
            "_train_one_batch_PINN",
            return_value=(0.0, None, len(trainer.train_dataset)),
        ):
            trainer.run_by_epoch_PINN()
        assert (
            trainer.num_training_samples_per_process
            == len(trainer.train_dataset)
        )

def main():
    world_size = 2
    epochs = 10
    mp.spawn(
        _run_apts_pinn,
        args=(world_size, epochs),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
