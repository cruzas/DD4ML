import copy
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import get_config, make_std_config
from .dist_utils import dprint
from .factory import criterion_factory, dataset_factory, optimizer_factory

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
    from dd4ml.pmw.model_handler import ModelHandler
    from dd4ml.pmw.parallelized_model import ParallelizedModel
    from dd4ml.trainer import Trainer

    # Select the source of configuration.
    config_src = wandb_config if wandb_config is not None else args
    dataset_name = config_src["dataset_name"]
    model_name = config_src["model_name"]
    optimizer_name = config_src["optimizer"]

    all_config = get_config(dataset_name, model_name, optimizer_name)
    if wandb_config is not None:
        all_config.merge_from_dict(wandb_config)
    else:
        args.pop("sweep_config", None)
    all_config.merge_from_dict(args)
    all_config.merge_and_cleanup(keys_to_look=["system", "model", "trainer"])

    # Instantiate dataset.
    dataset = dataset_factory.create(all_config.dataset_name, all_config.data)
    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False
    test_dataset = dataset.__class__(test_dataset_config)

    # Adjust config for text models (e.g., GPT).
    if hasattr(dataset, "vocab_size"):
        all_config.model.vocab_size = dataset.get_vocab_size()
    if hasattr(dataset, "block_size"):
        all_config.model.block_size = dataset.get_block_size()

    all_config = make_std_config(all_config)

    # Model instantiation (with optional parallelization).
    if getattr(all_config.trainer, "use_pmw", False) and hasattr(
        all_config.model.model_class, "as_model_dict"
    ):
        dprint("Using Parallel Model Wrapper and Model Handler")
        model_instance = all_config.model.model_class(all_config.model)
        model_dict = model_instance.as_model_dict()
        model_handler = ModelHandler(
            model_dict,
            all_config.model.num_subdomains,
            all_config.model.num_replicas_per_subdomain,
        )
        # if dist.get_rank() == 0: pprint.pprint(model_handler.nn_structure)
        all_config.trainer.model_handler = model_handler

        sample_input = dataset.get_sample_input(all_config.trainer)
        if sample_input.shape[0] == 1:
            other_sample = dataset.get_sample_input(all_config.trainer)
            sample_input = torch.cat([sample_input, other_sample], dim=0)

        model = ParallelizedModel(model_handler, sample=sample_input)
    else:
        model = all_config.model.model_class(all_config.model)
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if dist.get_backend() != "gloo"
            else "cpu"
        )
        model.to(device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            print(
                f"Rank {dist.get_rank()}, local rank {local_rank}, cuda available: {torch.cuda.is_available()}"
            )
            model = DDP(
                model, device_ids=[local_rank] if torch.cuda.is_available() else None
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
    lr = (
        wandb_config["learning_rate"]
        if wandb_config is not None
        else args["learning_rate"]
    )
    if optimizer_name in optimizer_factory.mapping:
        optimizer_obj = optimizer_factory.create(optimizer_name, model, lr)
        # Remove any unused attributes.
        for attr in [
            "subdomain_optimizer",
            "subdomain_optimizer_args",
            "global_optimizer",
            "global_optimizer_args",
            # TODO: probably need to update this
        ]:
            if hasattr(all_config.trainer, attr):
                delattr(all_config.trainer, attr)
    elif optimizer_name == "trust_region":
        from dd4ml.optimizers.trust_region import TrustRegion

        all_config.trainer = TrustRegion.setup_TR_args(all_config.trainer)
        optimizer_obj = TrustRegion(
            model=model,
            lr=all_config.trainer.learning_rate,
            max_lr=all_config.trainer.max_lr,
            min_lr=all_config.trainer.min_lr,
            nu=all_config.trainer.nu,
            inc_factor=all_config.trainer.inc_factor,
            dec_factor=all_config.trainer.dec_factor,
            nu_1=all_config.trainer.nu_1,
            nu_2=all_config.trainer.nu_2,
            max_iter=all_config.trainer.max_iter,
            norm_type=all_config.trainer.norm_type,
        )
    elif optimizer_name == "apts_ag":
        from dd4ml.optimizers.apts_ag import APTS_AG

        all_config.trainer = APTS_AG.setup_APTS_args(all_config.trainer)

        optimizer_obj = APTS_AG(
            model=model,
            subdomain_optimizer=all_config.trainer.subdomain_optimizer,
            subdomain_optimizer_defaults=all_config.trainer.subdomain_optimizer_args,
            global_optimizer=all_config.trainer.global_optimizer,
            global_optimizer_defaults=all_config.trainer.global_optimizer_args,
            lr=all_config.trainer.learning_rate,
            max_subdomain_iter=all_config.trainer.max_subdomain_iters,
            dogleg=False,
            APTS_in_data_sync_strategy="average",
            step_strategy="mean",
        )
    elif optimizer_name == "apts_d":
        from dd4ml.optimizers.apts_d import APTS_D

        all_config.trainer.norm_type = parse_norm(all_config.trainer.norm_type)

        all_config.trainer = APTS_D.setup_APTS_args(all_config.trainer)
        all_config.trainer.apts_d = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if dist.get_backend() != "gloo"
            else "cpu"
        )
        model.to(device)
        optimizer_obj = APTS_D(
            params=model.parameters(),
            model=model,
            criterion=criterion,
            device=device,
            nr_models=all_config.model.num_subdomains,
            global_opt=all_config.trainer.global_optimizer,
            global_opt_params=all_config.trainer.global_optimizer_args,
            local_opt=all_config.trainer.subdomain_optimizer,
            local_opt_params=all_config.trainer.subdomain_optimizer_args,
            global_pass=all_config.trainer.global_pass,
            foc=all_config.trainer.foc,
            correct_step=all_config.trainer.correct_step,
            norm_type=all_config.trainer.norm_type,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Set training schedule based on dataset.
    all_config.trainer.run_by_epoch = not ("shakespeare" in all_config.dataset_name)

    trainer = Trainer(
        all_config.trainer, model, optimizer_obj, criterion, dataset, test_dataset
    )
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
        and not "apts_ag" == wandb_config.get("optimizer", "").lower()
    ):
        args["use_pmw"] = False
        args["num_subdomains"] = dist.get_world_size() if dist.is_initialized() else 1

    config, _, trainer = get_config_model_and_trainer(args, wandb_config)
    dprint(config)

    if epoch_end_callback and trainer.config.run_by_epoch:
        trainer.set_callback("on_epoch_end", epoch_end_callback)
    if batch_end_callback and not trainer.config.run_by_epoch:
        trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()
