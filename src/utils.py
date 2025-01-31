import argparse
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.utility.dist_utils import *
from src.utility.mingpt_utils import *
from src.utility.ml_utils import *


def parse_cmd_args():
    parser = argparse.ArgumentParser("Hyperparameter Sweep")
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml", help="Sweep configuration file") 
    parser.add_argument("--entity", type=str, default="cruzaslocal", help="Wandb entity")
    parser.add_argument("--project", type=str, default="sgd_adam_hyperparameter_sweep", help="Wandb project")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--use_pmw", type=bool, default=True, help="Use Parallel Model Wrapper")
    parser.add_argument("--work_dir", type=str, default="../../saved_networks/wandb/", help="Directory to save models")
    # In case we are not executing with wandb sweep
    parser.add_argument("--dataset_name", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="simple_cnn", help="Model name")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--criterion", type=str, default="cross_entropy", help="Criterion name")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()
    return args 

def is_sweep_active(sweep_id, project, entity):
    """Check if the sweep is still running before using it."""
    api = wandb.Api()
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        return sweep.state == "running"  # Only use if it's still running
    except wandb.errors.CommError:
        return False  # Sweep doesn't exist

def generic_run(rank=None, master_addr=None, master_port=None, world_size=None, args=None, wandb_config=None, epoch_end_callback=None, batch_end_callback=None):
    use_wandb = wandb_config is not None  # Check if wandb should be used
    if use_wandb:
        wandb.init(config=wandb_config)
        wandb_config = wandb.config  # Assign updated config

    prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=torch.cuda.is_available())
    print(f"Rank {rank}/{world_size-1}")
    
    config, model, trainer = get_config_model_and_trainer(args, wandb_config)
    dprint(config)
    
    if epoch_end_callback is not None:
        trainer.set_callback("on_epoch_end", epoch_end_callback)
    if batch_end_callback is not None:
        trainer.set_callback("on_batch_end", batch_end_callback)
    
    # Run training
    trainer.run()
    
    # Destroy process group
    dist.destroy_process_group()
    
def get_config(dataset_name: str, model_name: str, optimizer: str = "sgd") -> CfgNode:
    from src.trainer import Trainer
    C = CfgNode()

    # System
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f'../../saved_networks/{dataset_name}/{model_name}/{optimizer}/'    

    # Data
    if dataset_name == "mnist":
        from src.datasets.mnist import MNISTDataset
        C.data = MNISTDataset.get_default_config()
    elif dataset_name == "cifar10":
        from src.datasets.cifar10 import CIFAR10Dataset
        C.data = CIFAR10Dataset.get_default_config()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Model
    if "cnn" in model_name.lower():
        # from src.models.cnn.my_cnn import MyCNN
        # C.model = MyCNN.get_default_config()
        # C.model.model_class = MyCNN
        
        from src.models.cnn.simple_cnn import SimpleCNN
        C.model = SimpleCNN.get_default_config()
        C.model.model_class = SimpleCNN
        
    # elif "resnet" in model_name.lower():
        # TODO
    
    C.model.input_channels = C.data.input_channels
    C.model.output_classes = C.data.output_classes

    # trainer
    C.trainer = Trainer.get_default_config()
    return C

def get_config_model_and_trainer(args, wandb_config):
    from src.pmw.model_handler import ModelHandler
    from src.pmw.parallelized_model import ParallelizedModel
    from src.trainer import Trainer
    
    if wandb_config is None:
        dataset_name = args["dataset_name"]
        model_name = args["model_name"]
        optimizer = args["optimizer"]
    else:
        dataset_name = wandb_config["dataset_name"]
        model_name = wandb_config["model_name"]
        optimizer = wandb_config["optimizer"]
    
    all_config = get_config(dataset_name, model_name, optimizer)
    if wandb_config is not None:
        all_config.merge_from_dict(wandb_config)
    else:
        if "sweep_config" in args:
            args.pop("sweep_config")
    all_config.merge_from_dict(args)
    all_config.merge_and_cleanup(keys_to_look=["system", "model", "trainer"])
    
    # Datasets
    if all_config.dataset_name == "mnist":
        from src.datasets.mnist import MNISTDataset
        dataset_class = MNISTDataset
    elif all_config.dataset_name == "cifar10":
        from src.datasets.cifar10 import CIFAR10Dataset
        dataset_class = CIFAR10Dataset
    else:
        raise ValueError(f"Unknown dataset name: {wandb_config['dataset_name']}")

    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False 
    
    train_dataset = dataset_class(all_config.data)
    test_dataset = dataset_class(test_dataset_config)
    
    # Define the model
    # Check if model_class has a method with build_*_dictionary 
    if hasattr(all_config.model.model_class, "as_model_dict"):
        # NOTE/REQUIRED: Regardless of the model class, it must define model_dict in this case.
        model = all_config.model.model_class(all_config.model)
        model_dict = model.as_model_dict()
        if all_config.trainer.use_pmw:
            print("Using Parallel Model Wrapper and Model Handler")
            model_handler = ModelHandler(model_dict, all_config.model.num_subdomains, all_config.model.num_replicas_per_subdomain)
            all_config.trainer.model_handler = model_handler
            
            # Construct the parallel model (overwrite the model)
            sample_input = train_dataset.get_sample_input(all_config.trainer)
            model = ParallelizedModel(model_handler, sample=sample_input)
        else:
            # Builds a standard PyTorch model based on a dictionary structure of its components
            from src.models.standard_model import build_standard_model
            model = build_standard_model(model_dict)
    else:
        model = all_config.model.model_class()
            
    dprint(model)
    
    # Define the criterion
    criterion_name = wandb_config["criterion"] if wandb_config is not None else args["criterion"]
    if criterion_name == "cross_entropy": 
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "weighted_cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=train_dataset.compute_class_weights())
    elif criterion_name == "mse":
        criterion = nn.MSELoss()
    elif criterion_name == "cross_entropy_transformers":
        criterion = cross_entropy_transformers
    else:
        raise ValueError(f"Unknown criterion: {wandb_config['criterion']}")
    
    # Define the optimizer
    lr = wandb_config["learning_rate"] if wandb_config is not None else args["learning_rate"]
    if all_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif all_config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif all_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif all_config.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif all_config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif all_config.optimizer == "apts":
        from src.optimizers.apts import APTS
        all_config = APTS.setup_APTS_args(all_config)
        optimizer = APTS(
                            model=model,
                            subdomain_optimizer=all_config.subdomain_optimizer,
                            subdomain_optimizer_defaults=all_config.subdomain_optimizer_args,
                            global_optimizer=all_config.global_optimizer,
                            global_optimizer_defaults=all_config.global_optimizer_args,
                            lr=all_config.learning_rate,
                            max_subdomain_iter=all_config.max_subdomain_iters,
                            dogleg=True,
                            APTS_in_data_sync_strategy='average', 
                            step_strategy='mean'
                        )
    
    trainer = Trainer(all_config.trainer, model, optimizer, criterion, train_dataset, test_dataset)
    
    return all_config, model, trainer
    

