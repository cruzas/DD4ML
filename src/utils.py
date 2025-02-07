import copy

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb

from src.utility.dist_utils import *
from src.utility.mingpt_utils import *
from src.utility.ml_utils import *
from src.utility.wandb_utils import *

# Global mapping dictionaries
DATASET_MAP = {
    "mnist": ("src.datasets.mnist", "MNISTDataset"),
    "cifar10": ("src.datasets.cifar10", "CIFAR10Dataset"),
    "tinyshakespeare": ("src.datasets.tinyshakespeare", "TinyShakespeareDataset")
}

MODEL_MAP = {
    "simple_cnn": ("src.models.cnn.simple_cnn", "SimpleCNN"),
    "big_cnn": ("src.models.cnn.big_cnn", "BigCNN"),
    "simple_resnet": ("src.models.resnet.simple_resnet", "SimpleResNet"),
    "mingpt": ("src.models.gpt.mingpt.model", "GPT")
}

CRITERION_MAP = {
    "cross_entropy": lambda train_ds=None: nn.CrossEntropyLoss(),
    "weighted_cross_entropy": lambda train_ds: nn.CrossEntropyLoss(weight=train_ds.compute_class_weights()),
    "mse": lambda train_ds=None: nn.MSELoss(),
    "cross_entropy_transformers": lambda train_ds=None: cross_entropy_transformers
}

OPTIMIZER_MAP = {
    "sgd": lambda model, lr: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
    "adam": lambda model, lr: torch.optim.Adam(model.parameters(), lr=lr),
    "adamw": lambda model, lr: torch.optim.AdamW(model.parameters(), lr=lr),
    "adagrad": lambda model, lr: torch.optim.Adagrad(model.parameters(), lr=lr),
    "rmsprop": lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
}


def broadcast_dict(d, src=0):
    """
    Broadcasts a dictionary from the source rank to all other ranks.
    Uses PyTorch's `dist.broadcast_object_list` to share data.
    """
    obj_list = [d] if dist.get_rank() == src else [None]  # Only source rank has the data
    dist.broadcast_object_list(obj_list, src=src)  # Broadcast
    return obj_list[0]  # Return the received dictionary


def import_attr(module_path: str, class_name: str):
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def generic_run(rank=None, master_addr=None, master_port=None, world_size=None,
                args=None, wandb_config=None, epoch_end_callback=None, batch_end_callback=None):
    
    use_wandb = wandb_config is not None

    if use_wandb:
        if rank == 0:  # Only rank 0 should initialize wandb
            if wandb.run is None:
                wandb.init(config=wandb_config)
            wandb_config = dict(wandb.config)  # Store the selected hyperparameters
        else:
            wandb_config = {}  # Other ranks wait for broadcast

        # Ensure all ranks use the same hyperparameters by broadcasting from rank 0
        wandb_config = broadcast_dict(wandb_config, src=0)

    else:
        wandb_config = {}

    # Load config, model, and trainer with synchronized hyperparameters
    config, model, trainer = get_config_model_and_trainer(args, wandb_config)
    dprint(config)

    if epoch_end_callback and trainer.config.run_by_epoch:
        trainer.set_callback("on_epoch_end", epoch_end_callback)
    if batch_end_callback and not trainer.config.run_by_epoch:
        trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()
    
def get_config(dataset_name: str, model_name: str, optimizer: str = "sgd") -> CfgNode:
    from src.trainer import Trainer
    C = CfgNode()

    # System configuration
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f'../../saved_networks/{dataset_name}/{model_name}/{optimizer}/'

    # Data configuration using DATASET_MAP
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    ds_module, ds_class_name = DATASET_MAP[dataset_name]
    dataset_cls = import_attr(ds_module, ds_class_name)
    C.data = dataset_cls.get_default_config()

    # Model configuration using MODEL_MAP
    key = next((k for k in MODEL_MAP if k in model_name.lower()), None)
    if key is None:
        raise ValueError(f"Unknown model name: {model_name}. Please implement it in src/models/")
    model_module, model_class_name = MODEL_MAP[key]
    model_cls = import_attr(model_module, model_class_name)
    C.model = model_cls.get_default_config()
    C.model.model_class = model_cls
    
    # For image-processing models
    if getattr(C.data, 'input_channels', None) is not None:
        C.model.input_channels = C.data.input_channels
    if getattr(C.data, 'input_height', None) is not None:
        C.model.input_height = C.data.input_height
    if getattr(C.data, 'input_width', None) is not None:
        C.model.input_width = C.data.input_width
    if getattr(C.data, 'output_classes', None) is not None:
        C.model.output_classes = C.data.output_classes    

    # Trainer configuration
    C.trainer = Trainer.get_default_config()
    return C


def get_config_model_and_trainer(args, wandb_config):
    from src.pmw.model_handler import ModelHandler
    from src.pmw.parallelized_model import ParallelizedModel
    from src.trainer import Trainer

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

    # Dataset instantiation using DATASET_MAP
    if all_config.dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset name: {all_config.dataset_name}. Please add it to ./src/datasets/")
    ds_module, ds_class_name = DATASET_MAP[all_config.dataset_name]
    dataset_cls = import_attr(ds_module, ds_class_name)

    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False

    train_dataset = dataset_cls(all_config.data)
    test_dataset = dataset_cls(test_dataset_config)

    # For GPT models
    if getattr(train_dataset, 'vocab_size', None) is not None:
        all_config.model.vocab_size = train_dataset.get_vocab_size()
    if getattr(train_dataset, 'block_size', None) is not None:
        all_config.model.block_size = train_dataset.get_block_size()

    # if getattr(all_config.model, 'n_layer', None) is not None:
    #     BaseModel.n_layer = all_config.model.n_layer

    # Model instantiation and optional parallelization
    if all_config.trainer.use_pmw and hasattr(all_config.model.model_class, "as_model_dict"):
        dprint("Using Parallel Model Wrapper and Model Handler")
        model_instance = all_config.model.model_class(all_config.model)
        model_dict = model_instance.as_model_dict()
        model_handler = ModelHandler(model_dict, all_config.model.num_subdomains,
                                     all_config.model.num_replicas_per_subdomain)
        all_config.trainer.model_handler = model_handler

        sample_input = train_dataset.get_sample_input(all_config.trainer)
        if sample_input.shape[0] == 1:
            other_sample = train_dataset.get_sample_input(all_config.trainer)
            sample_input = torch.cat([sample_input, other_sample], dim=0)
        model = ParallelizedModel(model_handler, sample=sample_input)
    else:
        model = all_config.model.model_class(all_config.model)

    dprint(model)

    # Criterion selection using CRITERION_MAP
    criterion_key = wandb_config["criterion"] if wandb_config is not None else args["criterion"]
    if criterion_key not in CRITERION_MAP:
        raise ValueError(f"Unknown criterion: {criterion_key}")
    criterion = CRITERION_MAP[criterion_key](train_dataset if "weighted" in criterion_key else None)

    # Optimizer selection using OPTIMIZER_MAP
    lr = wandb_config["learning_rate"] if wandb_config is not None else args["learning_rate"]
    if optimizer_name in OPTIMIZER_MAP:
        optimizer_obj = OPTIMIZER_MAP[optimizer_name](model, lr)
        # Check if all_config.trainer.subdomain_optimizer exists and if so remove it
        if hasattr(all_config.trainer, "subdomain_optimizer"):
            del all_config.trainer.subdomain_optimizer
        if hasattr(all_config.trainer, "subdomain_optimizer_args"):
            del all_config.trainer.subdomain_optimizer_args
        if hasattr(all_config.trainer, "global_optimizer"):
            del all_config.trainer.global_optimizer
        if hasattr(all_config.trainer, "global_optimizer_args"):
            del all_config.trainer.global_optimizer_args
    elif optimizer_name == "apts":
        from src.optimizers.apts import APTS
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
            APTS_in_data_sync_strategy='average', 
            step_strategy='mean'
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Check if all_config.dataset_name has "shakespeare" in it
    # Make sure to check for other text datasets in the future
    if "shakespeare" in all_config.dataset_name:
        all_config.trainer.run_by_epoch = False
    else:
        all_config.trainer.run_by_epoch = True

    trainer = Trainer(all_config.trainer, model, optimizer_obj, criterion, train_dataset, test_dataset)
    return all_config, model, trainer
