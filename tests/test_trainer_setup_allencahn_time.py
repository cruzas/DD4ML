import torch
from dd4ml.utility.trainer_setup import get_config_model_and_trainer


def test_trainer_setup_infers_input_features_allencahn_time():
    args = {
        "dataset_name": "allencahn1d_time",
        "model_name": "pinn_ffnn",
        "optimizer": "apts_pinn",
    }
    cfg, model, trainer = get_config_model_and_trainer(args, wandb_config=None)
    assert cfg.model.input_features == 2
