import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from src.utils import (broadcast_dict, detect_environment, dprint,
                       find_free_port, generic_run,
                       prepare_distributed_environment)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_cmd_args(APTS=True):
    parser = argparse.ArgumentParser("Running configuration file...")
    parser.add_argument("--entity", type=str, default="cruzaslocal", help="Wandb entity")
    parser.add_argument("--work_dir", type=str, default="../saved_networks/wandb/", help="Directory to save models")
    
    # Default parameters differ by APTS flag.
    defaults = {
        "sweep_config": "./config_files/config_apts.yaml" if APTS else "./config_files/config_sgd.yaml",
        "project": "apts_tests" if APTS else "sgd_hyperparameter_sweep",
        "num_stages": 2 if APTS else 1,
        "use_pmw": True if APTS else False,
        "num_subdomains": 2 if APTS else 1,
        "num_replicas_per_subdomain": 1
    }
    help_msgs = {
        "sweep_config": "Sweep configuration file",
        "project": "Wandb project",
        "num_stages": "Number of stages",
        "use_pmw": "Use Parallel Model Wrapper",
        "num_subdomains": "Number of subdomains",
        "num_replicas_per_subdomain": "Number of replicas per subdomain"
    }
    for arg, default in defaults.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default, help=help_msgs[arg])
    
    # Common arguments.
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--dataset_name", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="simple_cnn", help="Model name")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--criterion", type=str, default="cross_entropy", help="Criterion name")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--metric", type=str, choices=["loss", "accuracy"], default="loss",
                        help="Metric to determine best learning rate")
    parser.add_argument("--subdomain_optimizer", type=str, default="sgd", help="Subdomain optimizer")
    parser.add_argument("--global_optimizer", type=str, default="trust_region", help="Global optimizer")
    parser.add_argument("--max_subdomain_iters", type=int, default=3, help="Max iterations for subdomain optimizer")
    
    return parser.parse_args()

def main(rank, master_addr, master_port, world_size, args):
    if not dist.is_initialized():
        prepare_distributed_environment(rank=rank, master_addr=master_addr,
                                        master_port=master_port, world_size=world_size,
                                        is_cuda_enabled=torch.cuda.is_available())
    use_wandb = WANDB_AVAILABLE
    print(f"Rank {rank}/{world_size - 1} ready. Using wandb: {use_wandb}")
    
    wandb_config = {}
    if use_wandb and rank == 0:
        wandb.init(entity=args["entity"], project=args["project"])
        wandb_config = dict(wandb.config)
    wandb_config = broadcast_dict(wandb_config, src=0) if use_wandb else {}
    trial_args = {**args, **wandb_config}
    
    log_fn = dprint if not use_wandb else (lambda x: wandb.log(x))
    
    def epoch_end_callback(trainer, save_model=False, save_frequency=5):
        dprint(f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, "
               f"Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt:.2f}s")
        if rank == 0 and use_wandb:
            log_fn({
                "epoch": trainer.epoch_num,
                "epoch_time": trainer.epoch_dt,
                "loss": trainer.loss,
                "accuracy": trainer.accuracy,
                "running_time": trainer.running_time
            })
        if save_model and trainer.epoch_num % save_frequency == 0:
            dprint("Saving model...")
            proj = wandb_config.get("project", args["project"])
            model_path = os.path.join(args["work_dir"], f"model_{proj}_{trainer.epoch_num}.pt")
            os.makedirs(args["work_dir"], exist_ok=True)
            if args["use_pmw"]:
                trainer.model.save_state_dict(model_path)
            else:
                torch.save(trainer.model.state_dict(), model_path)
    
    def batch_end_callback(trainer):
        if rank == 0 and use_wandb:
            log_fn({
                "iter": trainer.iter_num,
                "loss": trainer.loss,
                "running_time": trainer.running_time
            })
        # if trainer.iter_num % 10 == 0:
        dprint(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss:.5f}")
    
    generic_run(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size,
                args=trial_args, wandb_config=wandb_config if use_wandb else None,
                epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback)

def run_local(args, sweep_config):
    master_addr = "localhost"
    master_port = find_free_port()
    world_size = (args["num_subdomains"] *
                  args["num_replicas_per_subdomain"] *
                  args["num_stages"])
    
    def spawn_training():
        mp.spawn(main, args=(master_addr, master_port, world_size, args),
                 nprocs=world_size, join=True)
    
    if WANDB_AVAILABLE:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args["project"])
        wandb.agent(sweep_id, function=spawn_training, count=None)
    else:
        spawn_training()

def run_cluster(args, sweep_config):
    prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None)
    rank = dist.get_rank()
    world_size = (args["num_subdomains"] *
                  args["num_replicas_per_subdomain"] *
                  args["num_stages"])
    
    if WANDB_AVAILABLE and rank == 0:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args["project"])
        wandb.agent(sweep_id,
                    function=lambda: main(rank, None, None, world_size, args),
                    count=None)
    else:
        wandb_config = broadcast_dict({}, src=0) if WANDB_AVAILABLE else {}
        main(rank, None, None, world_size, {**args, **wandb_config})

if __name__ == "__main__":
    args = vars(parse_cmd_args())
    with open(args["sweep_config"], "r") as f:
        sweep_config = yaml.safe_load(f)
    
    comp_env = detect_environment()
    for trial in range(args["trials"]):
        print(f"Starting trial {trial + 1}/{args['trials']}...")
        if comp_env == "local":
            print("Executing locally...")
            run_local(args, sweep_config)
        else:
            print("Executing on a cluster...")
            run_cluster(args, sweep_config)