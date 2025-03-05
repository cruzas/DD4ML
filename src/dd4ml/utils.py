import copy
import importlib
import pprint

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from dd4ml.utility.dist_utils import *
from dd4ml.utility.mingpt_utils import *
from dd4ml.utility.ml_utils import *

# Check whether wandb exists
try:
    from dd4ml.utility.wandb_utils import *
except ImportError:
    pass


# Utility: Dynamically import an attribute from a module.
def import_attr(module_path: str, attr: str):
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# Utility: Broadcast a dictionary using PyTorch's distributed utilities.
def broadcast_dict(d, src=0):
    obj_list = [d] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# Generic Factory for creating components.
class Factory:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def create(self, key: str, *args, **kwargs):
        if key not in self.mapping:
            raise ValueError(f"Unknown key: {key}")
        module_path, creator = self.mapping[key]
        if callable(creator):
            return creator(*args, **kwargs)
        cls = import_attr(module_path, creator)
        return cls(*args, **kwargs)


# Mapping definitions.
DATASET_MAP = {
    "mnist": ("dd4ml.datasets.mnist", "MNISTDataset"),
    "cifar10": ("dd4ml.datasets.cifar10", "CIFAR10Dataset"),
    "tinyshakespeare": ("dd4ml.datasets.tinyshakespeare", "TinyShakespeareDataset"),
}

MODEL_MAP = {
    "simple_cnn": ("dd4ml.models.cnn.simple_cnn", "SimpleCNN"),
    "big_cnn": ("dd4ml.models.cnn.big_cnn", "BigCNN"),
    "simple_resnet": ("dd4ml.models.resnet.simple_resnet", "SimpleResNet"),
    "mingpt": ("dd4ml.models.gpt.mingpt.model", "GPT"),
}

CRITERION_MAP = {
    "cross_entropy": ("", lambda ds=None: nn.CrossEntropyLoss()),
    "weighted_cross_entropy": (
        "",
        lambda ds: nn.CrossEntropyLoss(weight=ds.compute_class_weights()),
    ),
    "mse": ("", lambda ds=None: nn.MSELoss()),
    "cross_entropy_transformers": (
        "",
        lambda ds=None: cross_entropy_transformers,
    ),  # Assumes defined elsewhere.
}

OPTIMIZER_MAP = {
    "sgd": ("", lambda model, lr: optim.SGD(model.parameters(), lr=lr, momentum=0.9)),
    "adam": ("", lambda model, lr: optim.Adam(model.parameters(), lr=lr)),
    "adamw": ("", lambda model, lr: optim.AdamW(model.parameters(), lr=lr)),
    "adagrad": ("", lambda model, lr: optim.Adagrad(model.parameters(), lr=lr)),
    "rmsprop": ("", lambda model, lr: optim.RMSprop(model.parameters(), lr=lr)),
}

# Instantiate factories.
dataset_factory = Factory(DATASET_MAP)
model_factory = Factory(MODEL_MAP)
criterion_factory = Factory(CRITERION_MAP)
optimizer_factory = Factory(OPTIMIZER_MAP)


# Configuration cleanup: recursively remove keys.
def remove_keys(config, keys_to_remove):
    if isinstance(config, dict):
        for k in keys_to_remove:
            config.pop(k, None)
        for key, value in config.items():
            config[key] = remove_keys(value, keys_to_remove)
    elif hasattr(config, "__dict__"):
        for k in keys_to_remove:
            if k in config.__dict__:
                del config.__dict__[k]
        for k in list(config.__dict__.keys()):
            value = getattr(config, k)
            updated_value = remove_keys(value, keys_to_remove)
            if hasattr(config, k):  # Check if the attribute still exists
                setattr(config, k, updated_value)
    return config


# Standardize configuration by removing unnecessary keys.
def make_std_config(config):
    use_pmw = getattr(config.trainer, "use_pmw", False)
    if not use_pmw:
        keys_to_remove = ["num_stages", "num_replicas_per_subdomain", "model_handler"]
        if "apts_d" not in config.optimizer.lower():
            keys_to_remove.append("num_subdomains")
        config = remove_keys(config, keys_to_remove)
    if config.optimizer != "apts":
        keys_to_remove = [
            "subdomain_optimizer",
            "subdomain_optimizer_args",
            "global_optimizer",
            "global_optimizer_args",
        ]
        config = remove_keys(config, keys_to_remove)
    return config


# Refactored configuration creation.
def get_config(dataset_name: str, model_name: str, optimizer: str):
    from dd4ml.trainer import (
        Trainer,
    )  # Assumes Trainer and CfgNode are defined in dd4ml.trainer.

    C = CfgNode()  # CfgNode should be defined elsewhere.

    # System configuration.
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f"../../saved_networks/{dataset_name}/{model_name}/{optimizer}/"

    # Data configuration via DATASET_MAP.
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    ds_module, ds_class_name = DATASET_MAP[dataset_name]
    dataset_cls = import_attr(ds_module, ds_class_name)
    C.data = dataset_cls.get_default_config()

    # Model configuration via MODEL_MAP.
    key = next((k for k in MODEL_MAP if k in model_name.lower()), None)
    if key is None:
        raise ValueError(f"Unknown model name: {model_name}.")
    model_module, model_class_name = MODEL_MAP[key]
    model_cls = import_attr(model_module, model_class_name)
    C.model = model_cls.get_default_config()
    C.model.model_class = model_cls

    # Set image-specific properties if available.
    for attr in ["input_channels", "input_height", "input_width", "output_classes"]:
        if hasattr(C.data, attr):
            setattr(C.model, attr, getattr(C.data, attr))

    # Trainer configuration.
    C.trainer = Trainer.get_default_config()
    return C


# Refactored function to create config, model, and trainer.
def get_config_model_and_trainer(args, wandb_config):
    from dd4ml.pmw.model_handler import ModelHandler
    from dd4ml.pmw.parallelized_model import ParallelizedModel
    from dd4ml.trainer import Trainer

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

    # Dataset instantiation via factory.
    dataset = dataset_factory.create(all_config.dataset_name, all_config.data)
    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False
    test_dataset = dataset.__class__(test_dataset_config)

    # For GPT or similar models.
    if hasattr(dataset, "vocab_size"):
        all_config.model.vocab_size = dataset.get_vocab_size()
    if hasattr(dataset, "block_size"):
        all_config.model.block_size = dataset.get_block_size()

    all_config = make_std_config(all_config)

    # Model instantiation and optional parallelization.
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
        if dist.get_rank() == 0:
            pprint.pprint(model_handler.nn_structure)
        all_config.trainer.model_handler = model_handler

        sample_input = dataset.get_sample_input(all_config.trainer)
        if sample_input.shape[0] == 1:
            other_sample = dataset.get_sample_input(all_config.trainer)
            sample_input = torch.cat([sample_input, other_sample], dim=0)
            
        model = ParallelizedModel(model_handler, sample=sample_input)
    else:
        model = all_config.model.model_class(all_config.model)

    dprint(model)

    # Criterion selection.
    criterion_key = (
        wandb_config["criterion"] if wandb_config is not None else args["criterion"]
    )
    if criterion_key not in CRITERION_MAP:
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
    if optimizer_name in OPTIMIZER_MAP:
        optimizer_obj = optimizer_factory.create(optimizer_name, model, lr)
        for attr in [
            "subdomain_optimizer",
            "subdomain_optimizer_args",
            "global_optimizer",
            "global_optimizer_args",
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

    elif optimizer_name == "apts":
        from dd4ml.optimizers.apts import APTS

        all_config.trainer = APTS.setup_APTS_args(all_config.trainer)
        optimizer_obj = APTS(
            model=model,
            subdomain_optimizer=all_config.trainer.subdomain_optimizer,
            subdomain_optimizer_defaults=all_config.trainer.subdomain_optimizer_args,
            global_optimizer=all_config.trainer.global_optimizer,
            global_optimizer_defaults=all_config.trainer.global_optimizer_args,
            lr=all_config.trainer.learning_rate,
            max_subdomain_iter=all_config.trainer.max_subdomain_iters,
            dogleg=True,
            APTS_in_data_sync_strategy="average",
            step_strategy="mean",
        )
    elif optimizer_name == "apts_d":
        from dd4ml.optimizers.apts_d import APTS_D

        all_config.trainer = APTS_D.setup_APTS_args(all_config.trainer)
        all_config.trainer.apts_d = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if dist.get_backend() != "gloo"
            else "cpu"
        )
        model.to(device)
        model = DDP(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )
        optimizer_obj = APTS_D(
            params=model.parameters(),
            model=model,
            criterion=criterion,
            device=device,
            max_iter=3,
            nr_models=all_config.model.num_subdomains,
            global_opt=all_config.trainer.global_optimizer,
            global_opt_params=all_config.trainer.global_optimizer_args,
            local_opt=all_config.trainer.subdomain_optimizer,
            local_opt_params=all_config.trainer.subdomain_optimizer_args,
            global_pass=True,
            foc=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Set training schedule based on dataset.
    if "shakespeare" in all_config.dataset_name:
        all_config.trainer.run_by_epoch = False
    else:
        all_config.trainer.run_by_epoch = True

    trainer = Trainer(
        all_config.trainer, model, optimizer_obj, criterion, dataset, test_dataset
    )
    return all_config, model, trainer


# Entry point for running the experiment.
def generic_run(
    rank=None,
    args=None,
    wandb_config=None,
    epoch_end_callback=None,
    batch_end_callback=None,
):

    use_wandb = wandb_config is not None
    if use_wandb:
        if rank == 0:  # Only rank 0 initializes wandb.
            if wandb.run is None:
                wandb.init(config=wandb_config)
            wandb_config = dict(wandb.config)
        else:
            wandb_config = {}
        # Ensure all ranks use the same hyperparameters.
        wandb_config = broadcast_dict(wandb_config, src=0)
    else:
        wandb_config = {}

    if "apts_d" in wandb_config["optimizer"].lower():
        args["use_pmw"] = False
        args["num_subdomains"] = dist.get_world_size() if dist.is_initialized() else 1

    config, _, trainer = get_config_model_and_trainer(args, wandb_config)
    dprint(config)

    if epoch_end_callback and trainer.config.run_by_epoch:
        trainer.set_callback("on_epoch_end", epoch_end_callback)
    if batch_end_callback and not trainer.config.run_by_epoch:
        trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()
