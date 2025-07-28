from dd4ml.utility import generic_run


if __name__ == "__main__":
    args = {
        "dataset_name": "poisson3d",
        "model_name": "pinn_ffnn",
        "optimizer": "apts_d",
        "criterion": "pinn_poisson3d",
        "epochs": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_workers": 0,
    }
    generic_run(args=args, wandb_config=None)

