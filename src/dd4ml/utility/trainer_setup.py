import copy
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dd4ml.utility import (
    criterion_factory,
    dataset_factory,
    dprint,
    get_device,
    mark_trainable,
    optimizer_factory,
    print_params_norm,
)

from .config import GPT_MODEL_ALIASES, get_config, make_std_config

# You can now add new components dynamically at runtime by calling, e.g.:
# dataset_factory.register("new_dataset", "dd4ml.datasets.new_dataset", "NewDatasetClass")


def parse_norm(norm_value):
    if norm_value in ("inf", "Inf", "INF"):
        return float(math.inf)
    elif norm_value in ("-inf", "-Inf", "-INF"):
        return float(-math.inf)
    try:
        val = float(norm_value)
        return val
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported norm value: {norm_value}")


def get_config_model_and_trainer(args, wandb_config):
    """
    Create and return the standardized configuration, model, and trainer.
    """
    from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset
    from dd4ml.datasets.pinn_allencahn_time import AllenCahn1DTimeDataset
    from dd4ml.datasets.pinn_poisson import Poisson1DDataset
    from dd4ml.datasets.pinn_poisson2d import Poisson2DDataset
    from dd4ml.datasets.pinn_poisson3d import Poisson3DDataset
    from dd4ml.pmw.model_handler import ModelHandler
    from dd4ml.pmw.parallelized_model import ParallelizedModel
    from dd4ml.trainer import Trainer

    # Select the source of configuration.
    config_src = wandb_config if wandb_config is not None else args
    dataset_name = config_src["dataset_name"]
    model_name = config_src["model_name"]
    optimizer_name = config_src["optimizer"]

    if isinstance(optimizer_name, str):
        optimizer_name = optimizer_name.lower()
        config_src["optimizer"] = optimizer_name

    for opt_key in ("loc_opt", "glob_opt"):
        if opt_key in config_src and isinstance(config_src[opt_key], str):
            config_src[opt_key] = config_src[opt_key].lower()

    all_config = get_config(dataset_name, model_name, optimizer_name)

    if wandb_config is not None:
        all_config.merge_from_dict(wandb_config)
    else:
        args.pop("sweep_config", None)
    all_config.merge_from_dict(args)

    # Ensure the correct GPT model_type is set based on the provided
    model_key = next((k for k in GPT_MODEL_ALIASES if k in model_name.lower()), None)
    if model_key is not None and getattr(all_config.model, "model_type", None) is None:
        all_config.model.model_type = GPT_MODEL_ALIASES[model_key]

    # Allow `subdomain_opt` as an alias for `loc_opt` in configuration files
    if hasattr(all_config.trainer, "subdomain_opt") and not hasattr(
        all_config.trainer, "loc_opt"
    ):
        all_config.trainer.loc_opt = all_config.trainer.subdomain_opt
        delattr(all_config.trainer, "subdomain_opt")
    all_config.merge_and_cleanup(keys_to_look=["system", "data", "model", "trainer"])

    # Instantiate dataset.
    dataset = dataset_factory.create(all_config.dataset_name, all_config.data)
    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False
    test_dataset = dataset.__class__(test_dataset_config)

    if (
        getattr(all_config.trainer, "contiguous_subdomains", False)
        and all_config.trainer.num_subdomains > 1
        and hasattr(dataset, "split_domain")
        and optimizer_name != "apts_pinn"
    ):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        exclusive = getattr(all_config.trainer, "exclusive", True)
        train_splits = dataset.split_domain(world_size, exclusive=exclusive)
        test_splits = test_dataset.split_domain(world_size, exclusive=exclusive)
        dataset = train_splits[rank]
        test_dataset = test_splits[rank]

    # Automatically infer branch input dimension for models like DeepONet
    if getattr(all_config.model, "branch_input_dim", None) is None:
        if hasattr(dataset, "branch_data") and hasattr(dataset.branch_data, "shape"):
            all_config.model.branch_input_dim = dataset.branch_data.shape[1]
        elif hasattr(all_config.data, "n_sensors"):
            all_config.model.branch_input_dim = all_config.data.n_sensors

    # Adjust model input dimension for PINN datasets based on the dataset
    if (
        hasattr(all_config.model, "input_features")
        and all_config.model.input_features is not None
        and isinstance(
            dataset,
            (
                Poisson1DDataset,
                Poisson2DDataset,
                Poisson3DDataset,
                AllenCahn1DDataset,
                AllenCahn1DTimeDataset,
            ),
        )
    ):
        sample = dataset[0]
        if isinstance(dataset, AllenCahn1DTimeDataset):
            sample_x = torch.cat(sample[:2], dim=0)
        else:
            sample_x = sample[0]
        all_config.model.input_features = sample_x.numel()

    # Adjust config for text models (e.g., GPT).
    if hasattr(dataset, "vocab_size"):
        all_config.model.vocab_size = dataset.get_vocab_size()
    if hasattr(dataset, "block_size"):
        all_config.model.block_size = dataset.get_block_size()

    all_config = make_std_config(all_config)
    if hasattr(all_config.trainer, "norm_type"):
        all_config.trainer.norm_type = parse_norm(all_config.trainer.norm_type)

    # Cache frequently accessed config objects
    trainer_config = all_config.trainer
    model_config = all_config.model
    device = get_device()

    # Model instantiation (with optional parallelization).
    if getattr(trainer_config, "use_pmw", False) and hasattr(
        model_config.model_class, "as_model_dict"
    ):
        dprint("Using Parallel Model Wrapper and Model Handler")

        # Ensure num_stages is set in model_config
        if model_config.num_stages is None:
            raise ValueError(
                "model_config.num_stages is None. Please specify num_stages in your configuration."
            )

        print(f"Model config num_stages: {model_config.num_stages}")
        model_instance = model_config.model_class(model_config)
        model_dict = model_instance.as_model_dict()
        print(
            f"Creating ModelHandler with: num_subdomains={model_config.num_subdomains}, num_replicas_per_subdomain={model_config.num_replicas_per_subdomain}, num_stages={model_config.num_stages}"
        )
        print(
            f"Expected world_size should be: {model_config.num_subdomains * model_config.num_replicas_per_subdomain * model_config.num_stages}"
        )
        model_handler = ModelHandler(
            model_dict,
            model_config.num_subdomains,
            model_config.num_replicas_per_subdomain,
        )
        trainer_config.model_handler = model_handler

        sample_input = dataset.get_sample_input(trainer_config)
        if sample_input.shape[0] == 1:
            other_sample = dataset.get_sample_input(trainer_config)
            sample_input = torch.cat([sample_input, other_sample], dim=0)

        model = ParallelizedModel(model_handler, sample=sample_input)
    else:
        model = model_config.model_class(model_config)
        model.to(device)

        if dist.is_initialized() and dist.get_world_size() > 1:
            loc_rank = int(os.environ.get("LOCAL_RANK", 0))
            print(
                f"Rank {dist.get_rank()}, local rank {loc_rank}, cuda available: {torch.cuda.is_available()}"
            )
            model = DDP(
                model, device_ids=[loc_rank] if torch.cuda.is_available() else None
            )
    criterion_key = (
        wandb_config["criterion"] if wandb_config is not None else args["criterion"]
    )
    if criterion_key not in criterion_factory.mapping:
        raise ValueError(f"Unknown criterion: {criterion_key}")
    criterion = criterion_factory.create(
        criterion_key, dataset if "weighted" in criterion_key else None
    )

    # Optimizer selection.
    if optimizer_name in optimizer_factory.mapping:
        lr = (
            wandb_config["learning_rate"]
            if wandb_config is not None
            else args["learning_rate"]
        )

        optimizer_obj = optimizer_factory.create(optimizer_name, model, lr)
        # Remove any unused attributes.
        for attr in [
            "loc_opt",
            "loc_opt_hparams",
            "glob_opt",
            "glob_opt_hparams",
            # TODO: probably need to update this
        ]:
            if hasattr(all_config.trainer, attr):
                delattr(all_config.trainer, attr)
    elif optimizer_name == "tr":
        from dd4ml.optimizers.tr import TR

        all_config.trainer = TR.setup_TR_hparams(all_config.trainer)

        optimizer_obj = TR(
            params=model.parameters(),
            delta=all_config.trainer.delta,
            max_delta=all_config.trainer.max_delta,
            min_delta=all_config.trainer.min_delta,
            nu=all_config.trainer.nu,
            inc_factor=all_config.trainer.inc_factor,
            dec_factor=all_config.trainer.dec_factor,
            nu_dec=all_config.trainer.nu_dec,
            nu_inc=all_config.trainer.nu_inc,
            norm_type=all_config.trainer.norm_type,
            mem_length=all_config.trainer.mem_length,
            second_order=all_config.trainer.glob_second_order,
            dogleg=all_config.trainer.dogleg,
        )
    elif optimizer_name == "apts_ip":
        from dd4ml.optimizers.apts_ip import APTS_IP

        all_config.trainer = APTS_IP.setup_APTS_hparams(all_config.trainer)

        # print(f"APTS_IP got model: {model}")

        optimizer_obj = APTS_IP(
            params=model.parameters(),
            model=model,
            glob_opt=all_config.trainer.glob_opt,
            glob_opt_hparams=all_config.trainer.glob_opt_hparams,
            loc_opt=all_config.trainer.loc_opt,
            loc_opt_hparams=all_config.trainer.loc_opt_hparams,
            glob_pass=all_config.trainer.glob_pass,
            norm_type=all_config.trainer.norm_type,
            max_loc_iters=all_config.trainer.max_loc_iters,
            max_glob_iters=all_config.trainer.max_glob_iters,
            tol=all_config.trainer.tol,
            APTS_in_data_sync_strategy="average",
            step_strategy="mean",
            **all_config.trainer.apts_params,
        )
    elif optimizer_name == "apts_d":
        from dd4ml.optimizers.apts_d import APTS_D

        all_config.trainer = APTS_D.setup_APTS_hparams(all_config.trainer)
        all_config.trainer.apts_d = True

        optimizer_obj = APTS_D(
            params=model.parameters(),
            model=model,
            criterion=criterion,
            device=device,
            nr_models=model_config.num_subdomains,
            glob_opt=trainer_config.glob_opt,
            glob_opt_hparams=trainer_config.glob_opt_hparams,
            loc_opt=trainer_config.loc_opt,
            loc_opt_hparams=trainer_config.loc_opt_hparams,
            glob_pass=trainer_config.glob_pass,
            foc=trainer_config.foc,
            norm_type=trainer_config.norm_type,
            max_loc_iters=trainer_config.max_loc_iters,
            max_glob_iters=trainer_config.max_glob_iters,
            tol=trainer_config.tol,
            **trainer_config.apts_params,
        )
    elif optimizer_name == "apts_p":
        from dd4ml.optimizers.apts_p import APTS_P

        all_config.trainer = APTS_P.setup_APTS_hparams(all_config.trainer)
        loc_rank = int(os.environ.get("LOCAL_RANK", 0))

        optimizer_obj = APTS_P(
            params=model.parameters(),
            model=model,
            criterion=criterion,
            device=device,
            nr_models=model_config.num_subdomains,
            glob_opt=trainer_config.glob_opt,
            glob_opt_hparams=trainer_config.glob_opt_hparams,
            loc_opt=trainer_config.loc_opt,
            loc_opt_hparams=trainer_config.loc_opt_hparams,
            glob_pass=trainer_config.glob_pass,
            norm_type=trainer_config.norm_type,
            max_loc_iters=trainer_config.max_loc_iters,
            max_glob_iters=trainer_config.max_glob_iters,
            tol=trainer_config.tol,
            **trainer_config.apts_params,
        )
    elif optimizer_name == "apts_pinn":
        from dd4ml.optimizers.apts_pinn import APTS_PINN

        all_config.trainer = APTS_PINN.setup_APTS_hparams(all_config.trainer)

        optimizer_obj = APTS_PINN(
            params=model.parameters(),
            model=model,
            criterion=criterion,
            device=device,
            glob_opt=trainer_config.glob_opt,
            glob_opt_hparams=trainer_config.glob_opt_hparams,
            loc_opt=trainer_config.loc_opt,
            loc_opt_hparams=trainer_config.loc_opt_hparams,
            glob_pass=trainer_config.glob_pass,
            foc=trainer_config.foc,
            norm_type=trainer_config.norm_type,
            max_loc_iters=trainer_config.max_loc_iters,
            max_glob_iters=trainer_config.max_glob_iters,
            tol=trainer_config.tol,
            num_subdomains=trainer_config.num_subdomains,
            overlap=trainer_config.overlap,
            **trainer_config.apts_params,
        )
    elif optimizer_name == "lssr1_tr":
        from dd4ml.optimizers.lssr1_tr import LSSR1_TR

        all_config.trainer = LSSR1_TR.setup_LSSR1_TR_hparams(all_config.trainer)

        optimizer_obj = LSSR1_TR(
            params=model.parameters(),
            lr=all_config.trainer.learning_rate,
            delta=all_config.trainer.delta,
            min_delta=all_config.trainer.min_delta,
            max_delta=all_config.trainer.max_delta,
            gamma=all_config.trainer.gamma,
            second_order=all_config.trainer.glob_second_order,
            dogleg=all_config.trainer.dogleg,
            mem_length=all_config.trainer.mem_length,
            max_wolfe_iters=all_config.trainer.max_wolfe_iters,
            max_zoom_iters=all_config.trainer.max_zoom_iters,
            mu=all_config.trainer.mu,
            tau_1=all_config.trainer.tau_1,
            tau_2=all_config.trainer.tau_2,
            tau_3=all_config.trainer.tau_3,
            nu_1=all_config.trainer.nu_1,
            nu_2=all_config.trainer.nu_2,
            nu_3=all_config.trainer.nu_3,
            nu_4=all_config.trainer.nu_4,
            tol=all_config.trainer.tol,
            norm_type=all_config.trainer.norm_type,
            c_1=all_config.trainer.c_1,
            c_2=all_config.trainer.c_2,
            alpha_max=all_config.trainer.alpha_max,
            sync=True,
            paper_tr_update=all_config.trainer.paper_tr_update,
        )

    elif optimizer_name == "asntr":
        from dd4ml.optimizers.asntr import ASNTR

        all_config.trainer = ASNTR.setup_ASNTR_hparams(all_config.trainer)

        optimizer_obj = ASNTR(
            params=model.parameters(),
            device=device,
            lr=trainer_config.learning_rate,
            delta=trainer_config.delta,
            min_delta=trainer_config.min_delta,
            max_delta=trainer_config.max_delta,
            gamma=trainer_config.gamma,
            second_order=trainer_config.glob_second_order,
            dogleg=trainer_config.dogleg,
            mem_length=trainer_config.mem_length,
            eta=trainer_config.eta,
            nu=trainer_config.nu,
            eta_1=trainer_config.eta_1,
            eta_2=trainer_config.eta_2,
            tau_1=trainer_config.tau_1,
            tau_2=trainer_config.tau_2,
            tau_3=trainer_config.tau_3,
            norm_type=trainer_config.norm_type,
            c_1=trainer_config.c_1,
            c_2=trainer_config.c_2,
            alpha=trainer_config.alpha,
            tol=trainer_config.tol,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Set training schedule based on dataset.
    # all_config.trainer.run_by_epoch = not ("shakespeare" in all_config.dataset_name)

    # Check if cnn, ffnn, or resnet are substrings in the model name in lower case to determine whether to run by epoch or not
    if (
        "cnn" in all_config.model.model_class.__name__.lower()
        or "ffnn" in all_config.model.model_class.__name__.lower()
        or "resnet" in all_config.model.model_class.__name__.lower()
    ):
        all_config.trainer.run_by_epoch = True
    else:
        all_config.trainer.run_by_epoch = False

    if hasattr(all_config.model, "num_subdomains"):
        all_config.trainer.num_subdomains = all_config.model.num_subdomains
    else:
        all_config.model.num_subdomains = 1
    trainer = Trainer(
        all_config.trainer, model, optimizer_obj, criterion, dataset, test_dataset
    )
    trainer.optimizer.dataset_len = len(dataset)
    return all_config, model, trainer


def generic_run(
    rank=None,
    args=None,
    wandb_config=None,
    epoch_end_callback=None,
    batch_end_callback=None,
):
    """
    Entry point for running the experiment.
    """
    import wandb

    from .utils import broadcast_dict

    # Initialize wandb only on the root rank.
    use_wandb = wandb_config is not None
    if use_wandb:
        if rank == 0:
            if wandb.run is None:
                wandb.init(config=wandb_config)
            wandb_config = dict(wandb.config)
        else:
            wandb_config = {}
        wandb_config = broadcast_dict(wandb_config, src=0)
    else:
        wandb_config = {}

    # Adjust args for apts_d optimizer.
    if (
        "apts" in wandb_config.get("optimizer", "").lower()
        and not "apts_ip" == wandb_config.get("optimizer", "").lower()
    ):
        args["use_pmw"] = False
        args["num_subdomains"] = dist.get_world_size() if dist.is_initialized() else 1

    # Enable PMW for APTS_IP optimizer
    if wandb_config.get("optimizer", "").lower() == "apts_ip":
        args["use_pmw"] = True
        dprint(f"APTS_IP detected, setting use_pmw=True")

    config, _, trainer = get_config_model_and_trainer(args, wandb_config)
    dprint(config)
    dprint(f"Using device: {trainer.device}")

    if epoch_end_callback and trainer.config.run_by_epoch:
        trainer.set_callback("on_epoch_end", epoch_end_callback)
    if batch_end_callback and not trainer.config.run_by_epoch:
        trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()
