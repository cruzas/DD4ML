import argparse
import os
import pprint

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

def parse_cmd_args(APTS=False):
    parser = argparse.ArgumentParser("Running configuration file (or defaults if no wandb file is provided)...")
    parser.add_argument("--entity", type=str, default="cruzaslocal", help="Wandb entity")
    parser.add_argument("--work_dir", type=str, default="../saved_networks/wandb/", help="Directory to save models")
    
    if not APTS:
        parser.add_argument("--sweep_config", type=str, default="./config_files/config_sgd.yaml", help="Sweep configuration file") 
        parser.add_argument("--project", type=str, default="sgd_hyperparameter_sweep", help="Wandb project")
        parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
        parser.add_argument("--use_pmw", type=bool, default=False, help="Use Parallel Model Wrapper")
        parser.add_argument("--num_subdomains", type=int, default=1, help="Number of subdomains")
        parser.add_argument("--num_replicas_per_subdomain", type=int, default=1, help="Number of replicas per subdomain")
    elif APTS:
        parser.add_argument("--sweep_config", type=str, default="config_apts.yaml", help="Sweep configuration file") 
        parser.add_argument("--project", type=str, default="apts_tests", help="Wandb project")
        parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
        parser.add_argument("--use_pmw", type=bool, default=True, help="Use Parallel Model Wrapper")
        parser.add_argument("--num_subdomains", type=int, default=2, help="Number of subdomains")
        parser.add_argument("--num_replicas_per_subdomain", type=int, default=1, help="Number of replicas per subdomain")
    
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--dataset_name", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="simple_cnn", help="Model name")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--criterion", type=str, default="cross_entropy", help="Criterion name")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--metric", type=str, choices=["loss", "accuracy"], default="loss", help="Metric to determine best learning rate")
    parser.add_argument("--subdomain_optimizer", type=str, default="sgd", help="Subdomain optimizer")
    parser.add_argument("--global_optimizer", type=str, default="trust_region", help="Global optimizer")
    parser.add_argument("--max_subdomain_iters", type=int, default=3, help="Max iterations for subdomain optimizer")

    args = parser.parse_args()
    return args

def main(rank, master_addr, master_port, world_size, args):
    """Main training function with WandB sweeps."""
    
    # Ensure process group is initialized
    if not dist.is_initialized():
        prepare_distributed_environment(rank=rank, master_addr=master_addr, master_port=master_port, 
                                        world_size=world_size, is_cuda_enabled=torch.cuda.is_available())

    use_wandb = WANDB_AVAILABLE
    print(f"Rank {rank}/{world_size - 1} ready. Using wandb: {use_wandb}")

    wandb_config = {}
    if use_wandb and rank == 0:
        wandb.init(entity=args["entity"], project=args["project"])
        wandb_config = dict(wandb.config)

    # Broadcast wandb config to all processes so they get the same hyperparameters
    wandb_config = broadcast_dict(wandb_config, src=0) if use_wandb else {}
    
    # Merge WandB config into args
    trial_args = {**args, **wandb_config}

    logging_fn = dprint if not use_wandb else (lambda x: wandb.log(x))

    def epoch_end_callback(trainer, save_model=False, save_frequency=5):
        dprint(f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt:.2f}s")
        if rank == 0 and use_wandb:
            logging_fn({
                "epoch": trainer.epoch_num,
                "epoch_time": trainer.epoch_dt,
                "loss": trainer.loss,
                "accuracy": trainer.accuracy,
                "running_time": trainer.running_time
            })
        if save_model and trainer.epoch_num % save_frequency == 0:
            dprint("Saving model...")
            project_name = wandb_config.get("project", args["project"])
            model_path = os.path.join(args["work_dir"], f"model_{project_name}_{trainer.epoch_num}.pt")
            if not os.path.exists(args["work_dir"]):
                os.makedirs(args["work_dir"])
            if args["use_pmw"]:
                trainer.model.save_state_dict(model_path)
            else:
                torch.save(trainer.model.state_dict(), model_path)

    def batch_end_callback(trainer):
        if rank == 0 and use_wandb:
            logging_fn({
                "iter": trainer.iter_num,
                "loss": trainer.loss,
                "running_time": trainer.running_time
            })
        if trainer.iter_num % 10 == 0:
            dprint(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss:.5f}")

    # Run the training with the current hyperparameters
    generic_run(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size,
                args=trial_args, wandb_config=wandb_config if use_wandb else None,
                epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback)
    
if __name__ == "__main__":
    args = parse_cmd_args()
    args_dict = vars(args)

    # Load the sweep config
    with open(args_dict["sweep_config"], "r") as f:
        sweep_config = yaml.safe_load(f)

    # Detect if running locally or in a cluster
    comp_env = detect_environment()

    for trial in range(args.trials):
        print(f"Starting trial {trial + 1}/{args.trials}...")
        if comp_env == "local":
            print("Executing locally...")
            master_addr = "localhost"
            master_port = find_free_port()
            world_size = args.num_subdomains * args.num_replicas_per_subdomain * args.num_stages

            if WANDB_AVAILABLE:
                sweep_id = wandb.sweep(sweep=sweep_config, project=args_dict["project"])

                def wrapped_main():
                    mp.spawn(main, args=(master_addr, master_port, world_size, args_dict), nprocs=world_size, join=True)

                wandb.agent(sweep_id, function=wrapped_main, count=None)
            else:
                mp.spawn(main, args=(master_addr, master_port, world_size, args_dict), nprocs=world_size, join=True)

        else:
            print("Executing on a cluster...")

            # Extract SLURM rank information
            rank = None
            world_size = None
            master_addr = None
            master_port = None

            # Initialize the process group before calling dist.get_rank()
            prepare_distributed_environment(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size)
            rank = dist.get_rank()
            
            if WANDB_AVAILABLE and rank == 0:
                # Initialize the WandB sweep on the master rank
                sweep_id = wandb.sweep(sweep=sweep_config, project=args_dict["project"])

                def wrapped_main():
                    wandb.agent(sweep_id, function=lambda: main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args_dict), count=None)

                wrapped_main()
            else:
                # Wait for the master rank to get the sweep config, then broadcast
                wandb_config = {}  # Empty dict; rank 0 will fill it
                wandb_config = broadcast_dict(wandb_config, src=0) if WANDB_AVAILABLE else {}

                # Start training with received hyperparameters
                main(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args={**args_dict, **wandb_config})
            