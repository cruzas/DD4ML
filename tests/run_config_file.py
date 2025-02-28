import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from dd4ml.utils import (
    broadcast_dict,
    detect_environment,
    dprint,
    find_free_port,
    generic_run,
    prepare_distributed_environment,
    set_seed,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_cmd_args(APTS=False):
    parser = argparse.ArgumentParser("Running configuration file...")

    # Check if WANDB_MODE is set to 'online'
    wandb_entity_default = "cruzaslocal"
    if os.getenv("WANDB_MODE") == "online":
        wandb_entity_default = "cruzas-universit-della-svizzera-italiana"

    # Always-added arguments.
    parser.add_argument(
        "--entity", type=str, default=wandb_entity_default, help="Wandb entity"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="../saved_networks/wandb/",
        help="Directory to save models",
    )
    parser.add_argument(
        "--sweep_config",
        type=str,
        default=(
            "./config_files/config_apts.yaml"
            if APTS
            else "./config_files/config_sgd.yaml"
        ),
        help="Sweep configuration file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=("apts_tests" if APTS else "sgd_hyperparameter_sweep"),
        help="Wandb project",
    )
    parser.add_argument(
        "--use_pmw", type=bool, default=False, help="Use Parallel Model Wrapper"
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mnist", help="Dataset name"
    )
    parser.add_argument(
        "--model_name", type=str, default="simple_cnn", help="Model name"
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer name")
    parser.add_argument(
        "--criterion", type=str, default="cross_entropy", help="Criterion name"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss", "accuracy"],
        default="loss",
        help="Metric to determine best learning rate",
    )
    parser.add_argument(
        "--use_seed", type=bool, default=False, help="Use a seed for reproducibility. You probably don't want to do this for hyperparameter tuning."
    )


    # Preliminary parse to check conditions.
    args, _ = parser.parse_known_args()

    # Add APTS-related arguments only if "apts" is in the sweep_config string.
    if "apts" in args.sweep_config.lower():
        parser.add_argument(
            "--subdomain_optimizer", type=str, default="sgd", help="Subdomain optimizer"
        )
        parser.add_argument(
            "--global_optimizer",
            type=str,
            default="trust_region",
            help="Global optimizer",
        )
        parser.add_argument(
            "--max_subdomain_iters",
            type=int,
            default=3,
            help="Max iterations for subdomain optimizer",
        )

    # Add PMW-related arguments only if use_pmw is True.
    if args.use_pmw:
        parser.add_argument(
            "--num_stages",
            type=int,
            default=2,
            help="Number of stages",
        )
        parser.add_argument(
            "--num_subdomains", type=int, default=1, help="Number of subdomains"
        )
        parser.add_argument(
            "--num_replicas_per_subdomain",
            type=int,
            default=2,
            help="Number of replicas per subdomain",
        )

    return parser.parse_args()


def main(rank, master_addr, master_port, world_size, args):
    use_wandb = WANDB_AVAILABLE
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    comp_env = detect_environment()
    print(
        f"[main] Rank {rank}: SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    if comp_env != "local" and torch.cuda.is_available() and not dist.is_initialized():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        prepare_distributed_environment(
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            is_cuda_enabled=torch.cuda.is_available(),
        )
    else:
        print("[main] Process group already initialized. Skipping initialization...")
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Print rank, local rank, and current cuda device
    if torch.cuda.is_available():
        print(
            f"[main] Rank {rank}, local rank {local_rank}, cuda device {torch.cuda.current_device()}"
        )
        print(
            f"[main] Local rank {local_rank}, number of visible devices: {torch.cuda.device_count()}, seeing {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

    wandb_config = {}
    if use_wandb and rank == 0:
        wandb.init(entity=args["entity"], project=args["project"])
        wandb_config = dict(wandb.config)
    wandb_config = broadcast_dict(wandb_config, src=0) if use_wandb else {}
    
    # NOTE: Remove this for hyperparameter tuning
    if args["use_seed"]:
        set_seed(wandb_config.get("seed", 3407))
    
    trial_args = {**args, **wandb_config}

    log_fn = dprint if not use_wandb else (lambda x: wandb.log(x))

    def epoch_end_callback(trainer, save_model=False, save_frequency=5):
        dprint(
            f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, "
            f"Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt*1000:.2f}ms"
        )
        if rank == 0 and use_wandb:
            log_fn(
                {
                    "epoch": trainer.epoch_num,
                    "epoch_time": trainer.epoch_dt,
                    "loss": trainer.loss,
                    "accuracy": trainer.accuracy,
                    "running_time": trainer.running_time,
                }
            )
        if save_model and trainer.epoch_num % save_frequency == 0:
            dprint("Saving model...")
            proj = wandb_config.get("project", args["project"])
            model_path = os.path.join(
                args["work_dir"], f"model_{proj}_{trainer.epoch_num}.pt"
            )
            os.makedirs(args["work_dir"], exist_ok=True)
            if args["use_pmw"]:
                trainer.model.save_state_dict(model_path)
            else:
                torch.save(trainer.model.state_dict(), model_path)

    def batch_end_callback(trainer):
        if rank == 0 and use_wandb:
            log_fn(
                {
                    "iter": trainer.iter_num,
                    "loss": trainer.loss,
                    "running_time": trainer.running_time,
                }
            )
        # if trainer.iter_num % 10 == 0:
        dprint(
            f"iter_dt {trainer.iter_dt*1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss:.5f}"
        )

    generic_run(
        rank=rank,
        args=trial_args,
        wandb_config=wandb_config if use_wandb else None,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
    )


def run_local(args, sweep_config):
    master_addr = "localhost"
    master_port = find_free_port()
    world_size = 2
    if args["use_pmw"]:
        world_size = (
            args["num_subdomains"]
            * args["num_replicas_per_subdomain"]
            * args["num_stages"]
        )

    def spawn_training():
        mp.spawn(
            main,
            args=(master_addr, master_port, world_size, args),
            nprocs=world_size,
            join=True,
        )

    if WANDB_AVAILABLE:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args["project"])
        wandb.agent(sweep_id, function=spawn_training, count=None)
    else:
        spawn_training()


def run_cluster(args, sweep_config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    comp_env = detect_environment()
    if comp_env != "local" and torch.cuda.is_available() and not dist.is_initialized():
        torch.cuda.set_device(local_rank)

    prepare_distributed_environment(
        rank=None,
        master_addr=None,
        master_port=None,
        world_size=None,
        is_cuda_enabled=torch.cuda.is_available(),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args["use_pmw"]:
        assert world_size == (
            args["num_subdomains"]
            * args["num_replicas_per_subdomain"]
            * args["num_stages"]
        ), f"World size {world_size} does not match the number of subdomains {args['num_subdomains']}, replicas {args['num_replicas_per_subdomain']}, and stages {args['num_stages']} specified."
    if WANDB_AVAILABLE and rank == 0:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args["project"])
        print(f"[run_cluster] Rank {rank} calling wandb.agent...")
        wandb.agent(
            sweep_id,
            function=lambda: main(rank, None, None, world_size, args),
            count=None,
        )
        print(f"[run_cluster] Rank {rank} waiting at barrier...")
        try:
            dist.barrier()
            print(f"[run_cluster] Rank {rank} Exiting successfully...")
            sys.exit(0)
        except Exception as e:
            print(f"Barrier timeout: {e}. Aborting process group...")
            dist.destroy_process_group()
            sys.exit(1)
    else:
        print(f"[run_cluster] Rank {rank} running main function...")
        main(rank, None, None, world_size, args)
        print(f"[run_cluster] Rank {rank} waiting at barrier...")
        try:
            dist.barrier()
            print(f"[run_cluster] Rank {rank} Exiting successfully...")
            sys.exit(0)
        except Exception as e:
            print(f"Barrier timeout: {e}. Aborting process group...")
            dist.destroy_process_group()
            sys.exit(1)


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
