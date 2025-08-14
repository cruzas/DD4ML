import torch
from torch.utils.data import DataLoader

from dd4ml.datasets.pinn_allencahn_time import AllenCahn1DTimeDataset
from dd4ml.models.ffnn.pinn_ffnn import PINNFFNN
from dd4ml.trainer import Trainer
from dd4ml.utility.pinn_allencahn_time_loss import AllenCahnTimePINNLoss
from unittest.mock import patch


def test_allencahn_time_dataloader_batching():
    cfg = AllenCahn1DTimeDataset.get_default_config()
    ds = AllenCahn1DTimeDataset(cfg)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    x, t, flag = next(iter(loader))
    assert x.shape == (16, 1)
    assert t.shape == (16, 1)
    assert flag.shape == (16, 1)
    assert len(ds) > 16


@patch("dd4ml.trainer.dist.get_backend", return_value="gloo")
@patch("torch.distributed.is_initialized", return_value=False)
def test_trainer_with_allencahn_time_dataset(_mock_is_init, _mock_backend):
    # use a tiny dataset for quick test
    cfg_ds = AllenCahn1DTimeDataset.get_default_config()
    cfg_ds.nx_interior = 2
    cfg_ds.nt_interior = 2
    cfg_ds.n_boundary_t = 2
    cfg_ds.n_initial_x = 2
    ds = AllenCahn1DTimeDataset(cfg_ds)

    model_cfg = PINNFFNN.get_default_config()
    model_cfg.input_features = 2
    model_cfg.fc_layers = [4]
    model = PINNFFNN(model_cfg)

    criterion = AllenCahnTimePINNLoss()
    trainer_cfg = Trainer.get_default_config()
    trainer_cfg.epochs = 0
    trainer_cfg.batch_size = 4
    trainer_cfg.run_by_epoch = True
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = Trainer(trainer_cfg, model, optimizer, criterion, ds, ds)
    trainer.run()
