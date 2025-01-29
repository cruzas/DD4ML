import copy

import pandas as pd
import wandb

from src.utility.dist_utils import *
from src.utility.mingpt_utils import *
from src.utility.ml_utils import *


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
    
    
def get_config(dataset_name: str, model_name: str, optimizer: str = "sgd") -> CfgNode:
    from src.trainer_pmw import Trainer
    C = CfgNode()

    # System
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f'../../saved_networks/{dataset_name}/{model_name}/{optimizer}/'    

    # Model
    if "cnn" in model_name.lower():
        from src.models.cnn.my_cnn import MyCNN
        C.model = MyCNN.get_default_config()
        C.model.model_class = MyCNN
    # elif "resnet" in model_name.lower():
        # TODO
        
    # Data
    if dataset_name == "mnist":
        from src.datasets.mnist import MNISTDataset
        C.data = MNISTDataset.get_default_config()
    elif dataset_name == "cifar10":
        from src.datasets.cifar10 import CIFAR10Dataset
        C.data = CIFAR10Dataset.get_default_config()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    C.model.input_channels = C.data.input_channels
    C.model.output_classes = C.data.output_classes

    # trainer
    C.trainer = Trainer.get_default_config()
    return C

    
def get_config_model_and_trainer(args, config):
    from src.pmw.model_handler import ModelHandler
    from src.pmw.parallelized_model import ParallelizedModel
    from src.trainer_pmw import Trainer
    
    all_config = get_config(config['dataset_name'], config['model_name'], config['optimizer'])
    all_config.merge_from_dict(args)
    all_config.merge_and_cleanup(keys_to_look=["model", "trainer"])
    
    # Datasets
    if config['dataset_name'] == "mnist":
        from src.datasets.mnist import MNISTDataset
        dataset_class = MNISTDataset
    elif config['dataset_name'] == "cifar10":
        from src.datasets.cifar10 import CIFAR10Dataset
        dataset_class = CIFAR10Dataset
    else:
        raise ValueError(f"Unknown dataset name: {config['dataset_name']}")

    test_dataset_config = copy.deepcopy(all_config.data)
    test_dataset_config.train = False 
    
    train_dataset = dataset_class(all_config.data)
    test_dataset = dataset_class(test_dataset_config)
    
    # Define the model
    model = all_config.model.model_class(all_config.model)
    dprint(model)
    # NOTE: regardless of the model class, it must define model_dict. 
    model_handler = ModelHandler(model.model_dict, config.model)
    config.trainer.model_handler = model_handler
    
    # Construct the parallel model (overwrite the model)
    sample_input = train_dataset.get_sample_input(config.trainer)
    model = ParallelizedModel(model_handler, sample=sample_input)
    
    # Define the optimizer
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    
    return all_config, model, trainer
    

