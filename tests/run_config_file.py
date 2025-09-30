import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset
from dd4ml.datasets.pinn_poisson import Poisson1DDataset
from dd4ml.datasets.pinn_poisson2d import Poisson2DDataset
from dd4ml.datasets.pinn_poisson3d import Poisson3DDataset
from dd4ml.utility import (
    broadcast_dict,
    detect_environment,
    dprint,
    find_free_port,
    generic_run,
    is_main_process,
    prepare_distributed_environment,
    set_seed,
)

# torch.autograd.set_detect_anomaly(True)
# import warnings

# warnings.filterwarnings("error")


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Constants
DEFAULT_SEED = 3407
LOG_FREQUENCY = 5
SAVE_FREQUENCY_EPOCH = 5
SAVE_FREQUENCY_BATCH = 2000
BARRIER_TIMEOUT = 30


def get_optimizer_delta_info(optimizer):
    """Extract delta/learning rate info from optimizer."""
    if hasattr(optimizer, "delta"):
        return optimizer.delta, "delta"
    return optimizer.param_groups[0]["lr"], "lr"


def parse_cmd_args() -> argparse.Namespace:
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options] --entity ENTITY --project PROJECT ...",
        description="Parse command line arguments.",
    )

    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer name")
    parser.add_argument(
        "--tol", type=float, default=1e-6, help="Tolerance for convergence"
    )
    # Preliminary parse to determine defaults based on optimizer
    temp_args, _ = parser.parse_known_args()
    default_use_pmw = temp_args.optimizer == "apts_ip"

    default_config_file = f"./config_files/config_{temp_args.optimizer}.yaml"
    default_project = "debugging"

    parser.add_argument(
        "--use_pmw",
        action="store_true",
        default=default_use_pmw,
        help="Use Parallel Model Wrapper",
    )
    parser.add_argument(
        "--sweep_config",
        type=str,
        default=default_config_file,
        help="Sweep configuration file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=default_project,
        help="Wandb project",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="cruzas-universit-della-svizzera-italiana",
        help="Wandb entity",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="../saved_networks/wandb/",
        help="Directory to save models",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mnist", help="Dataset name"
    )
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap factor")
    parser.add_argument(
        "--model_name", type=str, default="simple_ffnn", help="Model name"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="Criterion name",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Trust-region radius")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss", "accuracy"],
        default="loss",
        help="Metric to determine best learning rate",
    )
    parser.add_argument(
        "--num_workers", type=int, default=num_cpus, help="Number of workers to use"
    )
    parser.add_argument(
        "--use_seed",
        action="store_true",
        default=False,
        help="Use a seed for reproducibility.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument(
        "--trial_num",
        type=int,
        default=0,
        help="Trial number. Used to generate seed for reproducibility.",
    )
    parser.add_argument(
        "--num_subdomains", type=int, default=1, help="Number of subdomains"
    )
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument(
        "--num_replicas_per_subdomain",
        type=int,
        default=1,
        help="Number of replicas per subdomain",
    )

    # Development branch: finalise parsing here
    return parser.parse_args()


def wait_and_exit(rank: int) -> None:
    """Wait at the barrier and exit gracefully."""
    try:
        if dist.is_initialized():
            dist.barrier(timeout=BARRIER_TIMEOUT)
        sys.exit(0)
    except Exception as e:
        print(f"Barrier timeout: {e}. Cleaning up...")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


def save_model_if_needed(
    trainer, count, frequency, work_dir, project, use_pmw, filename_template
):
    if count % frequency == 0 and count > 0:
        dprint("Saving model...")
        model_path = os.path.join(
            work_dir, filename_template.format(project=project, count=count)
        )
        os.makedirs(work_dir, exist_ok=True)
        if use_pmw:
            trainer.model.save_state_dict(model_path)
        else:
            torch.save(trainer.model.state_dict(), model_path)


def main(
    rank: int, master_addr: str, master_port: str, world_size: int, args: dict
) -> None:
    """Main training routine executed by each process."""
    use_wandb = WANDB_AVAILABLE
    loc_rank = int(os.environ.get("SLURM_LOCALID", 0))
    comp_env = detect_environment()
    if comp_env != "local" and torch.cuda.is_available() and not dist.is_initialized():
        torch.cuda.set_device(loc_rank)

    if not dist.is_initialized():
        prepare_distributed_environment(
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            is_cuda_enabled=torch.cuda.is_available(),
        )

    rank = dist.get_rank() if dist.is_initialized() else 0

    wandb_config = {}
    if use_wandb and rank == 0:
        wandb.init(entity=args["entity"], project=args["project"])
        wandb_config = dict(wandb.config)
    wandb_config = broadcast_dict(wandb_config, src=0) if use_wandb else {}

    trial_args = {**args, **wandb_config}
    if trial_args["use_seed"]:
        trial_num = trial_args["trial_num"]
        seed = wandb_config.get("seed", DEFAULT_SEED) * trial_num
        set_seed(seed)

    apts_id = f"nst_{trial_args['num_stages']}_nsd_{trial_args['num_subdomains']}_nrpsd_{trial_args['num_replicas_per_subdomain']}"
    if trial_args["use_seed"]:
        apts_id += str(seed)

    log_fn = wandb.log if (use_wandb and rank == 0) else dprint

    def epoch_end_callback(
        trainer, save_model: bool = False, save_frequency: int = SAVE_FREQUENCY_EPOCH
    ) -> None:

        delta, thing_to_print = get_optimizer_delta_info(trainer.optimizer)

        if isinstance(
            trainer.train_dataset,
            (Poisson1DDataset, Poisson2DDataset, Poisson3DDataset, AllenCahn1DDataset),
        ):
            dprint(
                f"Epoch {trainer.epoch_num}, g-evals: {trainer.grad_evals}, loss: {trainer.loss:.4e}, running time: {trainer.running_time:.2f}s, {thing_to_print}: {delta:.6e}"
            )
        else:
            dprint(
                f"Epoch {trainer.epoch_num}, g-evals: {trainer.grad_evals}, loss: {trainer.loss:.4f}, accuracy: {trainer.accuracy:.2f}%, running time: {trainer.running_time:.2f}s, {thing_to_print}: {delta:.6e}"
            )

        if rank == 0 and use_wandb:
            log_fn(
                {
                    "epoch": trainer.epoch_num,
                    "epoch_time": trainer.epoch_dt,
                    "loss": trainer.loss,
                    "accuracy": trainer.accuracy,
                    "running_time": trainer.running_time,
                    "grad_evals": trainer.grad_evals,
                    f"{thing_to_print}": delta,
                    "batch_size": trainer.current_batch_size,
                }
            )
        if save_model:
            proj = wandb_config.get("project", trial_args["project"])
            save_model_if_needed(
                trainer,
                count=trainer.epoch_num,
                frequency=save_frequency,
                work_dir=trial_args["work_dir"],
                project=proj,
                use_pmw=trial_args["use_pmw"],
                filename_template="model_{project}_{count}.pt",
            )

    def batch_end_callback(
        trainer, save_model: bool = True, save_frequency: int = SAVE_FREQUENCY_BATCH
    ) -> None:
        delta, thing_to_print = get_optimizer_delta_info(trainer.optimizer)

        if trainer.iter_num % LOG_FREQUENCY == 0:
            if rank == 0 and use_wandb:
                log_fn(
                    {
                        "iter": trainer.iter_num,
                        "loss": trainer.loss,
                        "train_perplexity": trainer.curr_train_perplexity,
                        "running_time": trainer.running_time,
                        "grad_evals": trainer.grad_evals,
                        f"{thing_to_print}": delta,
                        "batch_size": trainer.current_batch_size,
                    }
                )

            dprint(
                f"Iter {trainer.iter_num}, g-evals: {trainer.grad_evals}, time {trainer.iter_dt * 1000:.2f}ms, running time: {trainer.running_time:.2f}s, loss {trainer.loss:.5f}, train perplexity: {trainer.curr_train_perplexity:.5f}, {thing_to_print}: {delta:.6e}"
            )
            if save_model:
                proj = wandb_config.get("project", trial_args["project"])
                filename = f"{proj}_{apts_id}_iter_{{count}}.pt"

                save_model_if_needed(
                    trainer,
                    count=trainer.iter_num,
                    frequency=save_frequency,
                    work_dir=trial_args["work_dir"],
                    project=proj,
                    use_pmw=trial_args["use_pmw"],
                    filename_template=filename,
                )

    generic_run(
        rank=rank,
        args=trial_args,
        wandb_config=wandb_config if use_wandb else None,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
    )


def run_local(args: dict, sweep_config: dict) -> None:
    master_addr = "localhost"
    master_port = find_free_port()

    # Extract config values from sweep_config if available and merge into args
    # This ensures world_size is calculated using config file values, not command-line defaults
    if sweep_config and "parameters" in sweep_config:
        for key, value_dict in sweep_config["parameters"].items():
            if "value" in value_dict:
                # Update args with config file values
                args[key] = value_dict["value"]

    # Always calculate world_size based on config values
    world_size = (
        args["num_subdomains"]
        * args["num_replicas_per_subdomain"]
        * args["num_stages"]
    )
    if args["use_pmw"]:
        print(
            f"PMW enabled: world_size = {args['num_subdomains']} * {args['num_replicas_per_subdomain']} * {args['num_stages']} = {world_size}"
        )
    else:
        print(f"PMW disabled: world_size = {args['num_subdomains']} * {args['num_replicas_per_subdomain']} * {args['num_stages']} = {world_size}")

    def spawn_training() -> None:
        if world_size == 1:
            # For single process, run directly without multiprocessing
            print("Running single process without multiprocessing")
            main(0, master_addr, master_port, world_size, args)
        else:
            # Use multiprocessing for multiple processes
            print(f"Running {world_size} processes with multiprocessing")
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


def run_cluster(args: dict, sweep_config: dict) -> None:
    loc_rank = int(os.environ.get("LOCAL_RANK", 0))
    comp_env = detect_environment()
    if comp_env != "local" and torch.cuda.is_available() and not dist.is_initialized():
        torch.cuda.set_device(loc_rank)

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
        expected_ws = (
            args["num_subdomains"]
            * args["num_replicas_per_subdomain"]
            * args["num_stages"]
        )
        assert world_size == expected_ws, (
            f"World size {world_size} does not match expected configuration: {expected_ws} "
            f"(subdomains: {args['num_subdomains']}, replicas: {args['num_replicas_per_subdomain']}, stages: {args['num_stages']})."
        )

    if WANDB_AVAILABLE and rank == 0:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args["project"])
        wandb.agent(
            sweep_id,
            function=lambda: main(rank, None, None, world_size, args),
            count=None,
        )
    else:
        main(rank, None, None, world_size, args)

    wait_and_exit(rank)


def main_single_run():
    """Run a single experiment with command line arguments."""
    args = vars(parse_cmd_args())
    try:
        with open(args["sweep_config"], "r") as f:
            sweep_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args['sweep_config']} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {args['sweep_config']}: {e}")
        sys.exit(1)

    comp_env = detect_environment()
    if comp_env == "local":
        run_local(args, sweep_config)
    else:
        run_cluster(args, sweep_config)


if __name__ == "__main__":
    main_single_run()
