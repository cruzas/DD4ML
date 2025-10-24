#!/usr/bin/env python3
"""
Hyperparameterization Test: SGD vs APTS_D for CNNs

This script tests the hypothesis that SGD performs worse than APTS_D on
overhyperparameterized networks. It runs both optimizers on MNIST with
increasingly complex CNN architectures to demonstrate where SGD struggles.

Usage:
    python hyperparameterization_test_cnn.py
    python hyperparameterization_test_cnn.py --num-conv-layers 2 4 8
    python hyperparameterization_test_cnn.py --filters 32 64 128
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml
from experiment_tracker import ExperimentTracker


def create_config(
    optimizer: str,
    filters: int,
    num_conv_layers: int,
    base_dir: Path,
    epochs: int = 2,
    batch_size: int = 10000,
    overlap: float = 0.0,
    batch_inc_factor: float = 1.0,
    max_wolfe_iters: int = None,
    max_zoom_iters: int = None,
    trial: int = 1,
    num_subdomains: int = None,
    num_replicas_per_subdomain: int = None,
    num_stages: int = None,
    fc_width: int = 128,
    pool_every: int = 2,
) -> Path:
    """Create a configuration file for the given optimizer and network size."""

    # Select base config template
    base_config = base_dir / "config_files" / f"config_{optimizer}.yaml"

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    # Load base config
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Update parameters
    config["parameters"]["dataset_name"]["value"] = "mnist"
    config["parameters"]["model_name"]["value"] = "medium_cnn"

    # CNN-specific parameters
    if "num_conv_layers" not in config["parameters"]:
        config["parameters"]["num_conv_layers"] = {}
    config["parameters"]["num_conv_layers"]["value"] = num_conv_layers

    if "filters_per_layer" not in config["parameters"]:
        config["parameters"]["filters_per_layer"] = {}
    config["parameters"]["filters_per_layer"]["value"] = filters

    if "fc_width" not in config["parameters"]:
        config["parameters"]["fc_width"] = {}
    config["parameters"]["fc_width"]["value"] = fc_width

    if "pool_every" not in config["parameters"]:
        config["parameters"]["pool_every"] = {}
    config["parameters"]["pool_every"]["value"] = pool_every

    # Common parameters
    config["parameters"]["batch_size"]["value"] = batch_size
    config["parameters"]["effective_batch_size"]["value"] = batch_size
    config["parameters"]["epochs"]["value"] = epochs
    config["parameters"]["max_iters"]["value"] = 0
    config["parameters"]["criterion"]["value"] = "cross_entropy"
    config["parameters"]["seed"]["value"] = 42
    config["parameters"]["shuffle"]["value"] = False

    # Add trial parameter if it doesn't exist
    if "trial" not in config["parameters"]:
        config["parameters"]["trial"] = {}
    config["parameters"]["trial"]["value"] = trial

    # Optimizer-specific settings
    if optimizer == "sgd":
        config["parameters"]["learning_rate"]["value"] = 0.01
        config["parameters"]["num_subdomains"]["value"] = (
            num_subdomains if num_subdomains is not None else 1
        )
        config["parameters"]["overlap"]["value"] = overlap
        config["parameters"]["batch_inc_factor"]["value"] = batch_inc_factor
    elif optimizer in ["apts_d", "apts_p"]:
        config["parameters"]["num_subdomains"]["value"] = (
            num_subdomains if num_subdomains is not None else 2
        )
        config["parameters"]["max_loc_iters"]["value"] = 2
        config["parameters"]["glob_second_order"]["value"] = False
        config["parameters"]["loc_second_order"]["value"] = False
        config["parameters"]["glob_dogleg"]["value"] = False
        config["parameters"]["loc_dogleg"]["value"] = False
        config["parameters"]["overlap"]["value"] = overlap
        config["parameters"]["batch_inc_factor"]["value"] = batch_inc_factor
        if max_wolfe_iters is not None:
            config["parameters"]["max_wolfe_iters"]["value"] = max_wolfe_iters
        if max_zoom_iters is not None:
            config["parameters"]["max_zoom_iters"]["value"] = max_zoom_iters
    elif optimizer == "apts_ip":
        config["parameters"]["num_stages"]["value"] = (
            num_stages if num_stages is not None else 2
        )
        config["parameters"]["max_loc_iters"]["value"] = 2
        config["parameters"]["glob_second_order"]["value"] = False
        config["parameters"]["loc_second_order"]["value"] = False
        config["parameters"]["glob_dogleg"]["value"] = False
        config["parameters"]["loc_dogleg"]["value"] = False
        config["parameters"]["overlap"]["value"] = overlap
        config["parameters"]["batch_inc_factor"]["value"] = batch_inc_factor
        if max_wolfe_iters is not None:
            config["parameters"]["max_wolfe_iters"]["value"] = max_wolfe_iters
        if max_zoom_iters is not None:
            config["parameters"]["max_zoom_iters"]["value"] = max_zoom_iters

    # Set num_replicas_per_subdomain if provided (applies to all optimizers)
    if num_replicas_per_subdomain is not None:
        config["parameters"]["num_replicas_per_subdomain"][
            "value"
        ] = num_replicas_per_subdomain

    # Create output config file
    config_name = f"config_hyperparam_cnn_{optimizer}_f{filters}_cl{num_conv_layers}_trial{trial}.yaml"
    output_path = base_dir / "config_files" / config_name

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def check_experiment_completed(config_path: Path, tracker: ExperimentTracker) -> bool:
    """Check if an experiment has already been completed successfully."""
    return tracker.is_completed(config_path=config_path)


def run_experiment(
    config_path: Path,
    test_dir: Path,
    tracker: ExperimentTracker,
    skip_existing: bool = False,
    metadata: dict = None,
) -> Tuple[bool, str]:
    """Run a single experiment with the given config."""
    job_name = config_path.stem

    # Check if experiment already completed
    if skip_existing and check_experiment_completed(config_path, tracker):
        print(f"  ⏭ Skipped (already completed): {job_name}")
        return True, "skipped"

    print(f"  Running: {job_name}")

    try:
        # Save current directory
        original_cwd = os.getcwd()

        # Change to test directory
        os.chdir(test_dir)

        # Run the experiment
        result = subprocess.run(
            [sys.executable, "run_config_file.py", "--sweep_config", str(config_path)],
            capture_output=False,
            text=True,
            timeout=2400,  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"  ✓ Completed: {job_name}")
            # Mark experiment as completed in tracker
            tracker.mark_completed(config_path=config_path, metadata=metadata)
            return True, ""
        else:
            print(f"  ✗ Failed: {job_name}")
            tracker.mark_failed(config_path=config_path, error="Non-zero return code")
            return False, ""

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {job_name}")
        tracker.mark_failed(config_path=config_path, error="Experiment timed out")
        return False, "Experiment timed out"
    except Exception as e:
        print(f"  ✗ Error: {job_name}: {e}")
        tracker.mark_failed(config_path=config_path, error=str(e))
        return False, str(e)
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Test hyperparameterization effects on SGD vs APTS_D for CNNs"
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        type=int,
        default=[32],
        help="Number of filters per layer (default: 32)",
    )
    parser.add_argument(
        "--num-conv-layers",
        nargs="+",
        type=int,
        default=[2],
        help="Number of convolutional layers to test (default: 2 4 8)",
    )
    parser.add_argument(
        "--fc-width",
        type=int,
        default=128,
        help="Width of fully connected layer (default: 128)",
    )
    parser.add_argument(
        "--pool-every",
        type=int,
        default=2,
        help="Apply pooling every N layers (default: 2)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        type=str,
        default=["sgd", "apts_d"],
        help="Optimizers to test: sgd, apts_d, apts_p, apts_ip (default: sgd apts_d)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size (default: 10000)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap parameter for optimizers (default: 0.0)",
    )
    parser.add_argument(
        "--batch-inc-factor",
        type=float,
        default=1.0,
        help="Batch increase factor for optimizers (default: 1.0)",
    )
    parser.add_argument(
        "--max-wolfe-iters",
        type=int,
        default=None,
        help="Maximum Wolfe line search iterations (optional, uses config default if not set)",
    )
    parser.add_argument(
        "--max-zoom-iters",
        type=int,
        default=None,
        help="Maximum zoom iterations for line search (optional, uses config default if not set)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run for each experiment (default: 1)",
    )
    parser.add_argument(
        "--num-subdomains",
        type=int,
        default=None,
        help="Number of subdomains (optional, uses optimizer defaults if not set: SGD=1, APTS_D/P=2)",
    )
    parser.add_argument(
        "--num-replicas-per-subdomain",
        type=int,
        default=None,
        help="Number of replicas per subdomain (optional, uses config default if not set)",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=None,
        help="Number of stages for APTS_IP (optional, uses default=2 if not set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create configs but don't run experiments",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated config files before starting",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that have already completed successfully",
    )

    args = parser.parse_args()

    # Find test directory
    script_dir = Path(__file__).parent

    # Initialize experiment tracker
    tracker = ExperimentTracker(script_dir / ".experiment_tracker_cnn.json")

    print("=" * 80)
    print("HYPERPARAMETERIZATION TEST: SGD vs APTS_D for CNNs")
    print("=" * 80)
    print(f"\nTest configuration:")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Filters per layer: {args.filters}")
    print(f"  Number of conv layers: {args.num_conv_layers}")
    print(f"  FC width: {args.fc_width}")
    print(f"  Pool every: {args.pool_every} layers")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Overlap: {args.overlap}")
    print(f"  Batch increase factor: {args.batch_inc_factor}")
    if args.max_wolfe_iters is not None:
        print(f"  Max Wolfe iters: {args.max_wolfe_iters}")
    if args.max_zoom_iters is not None:
        print(f"  Max zoom iters: {args.max_zoom_iters}")
    print(f"  Number of trials: {args.num_trials}")
    if args.num_subdomains is not None:
        print(f"  Num subdomains: {args.num_subdomains}")
    if args.num_replicas_per_subdomain is not None:
        print(f"  Num replicas per subdomain: {args.num_replicas_per_subdomain}")
    if args.num_stages is not None:
        print(f"  Num stages: {args.num_stages}")
    print(f"  Skip existing: {args.skip_existing}")
    if args.skip_existing:
        print(f"  Previously completed: {tracker.get_completed_count()}")
    print()

    # Clean old configs if requested
    if args.clean:
        print("Cleaning old config files...")
        config_dir = script_dir / "config_files"
        for config_file in config_dir.glob("config_hyperparam_cnn_*.yaml"):
            config_file.unlink()
            print(f"  Removed: {config_file.name}")
        print()

    # Generate all configurations
    configs = []
    total_experiments = (
        len(args.optimizers)
        * len(args.filters)
        * len(args.num_conv_layers)
        * args.num_trials
    )

    print(f"Generating {total_experiments} configurations...")
    for optimizer in args.optimizers:
        for filters in args.filters:
            for num_conv_layers in args.num_conv_layers:
                for trial in range(1, args.num_trials + 1):
                    try:
                        config_path = create_config(
                            optimizer=optimizer,
                            filters=filters,
                            num_conv_layers=num_conv_layers,
                            base_dir=script_dir,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            overlap=args.overlap,
                            batch_inc_factor=args.batch_inc_factor,
                            max_wolfe_iters=args.max_wolfe_iters,
                            max_zoom_iters=args.max_zoom_iters,
                            trial=trial,
                            num_subdomains=args.num_subdomains,
                            num_replicas_per_subdomain=args.num_replicas_per_subdomain,
                            num_stages=args.num_stages,
                            fc_width=args.fc_width,
                            pool_every=args.pool_every,
                        )
                        configs.append(
                            (optimizer, filters, num_conv_layers, trial, config_path)
                        )
                        print(f"  ✓ Created: {config_path.name}")
                    except Exception as e:
                        print(
                            f"  ✗ Failed to create config for {optimizer} f={filters} cl={num_conv_layers} trial={trial}: {e}"
                        )

    print(f"\nGenerated {len(configs)} configurations")

    if args.dry_run:
        print("\n[DRY RUN] - Not running experiments")
        return

    # Run experiments
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80 + "\n")

    results = []
    for i, (optimizer, filters, num_conv_layers, trial, config_path) in enumerate(
        configs, 1
    ):
        print(
            f"[{i}/{len(configs)}] {optimizer} - Filters: {filters}, Conv Layers: {num_conv_layers}, Trial: {trial}"
        )
        metadata = {
            "optimizer": optimizer,
            "filters": filters,
            "num_conv_layers": num_conv_layers,
            "trial": trial,
            "fc_width": args.fc_width,
            "pool_every": args.pool_every,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        success, output = run_experiment(
            config_path,
            script_dir,
            tracker,
            skip_existing=args.skip_existing,
            metadata=metadata,
        )
        results.append(
            {
                "optimizer": optimizer,
                "filters": filters,
                "num_conv_layers": num_conv_layers,
                "trial": trial,
                "success": success,
                "skipped": output == "skipped",
            }
        )
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r.get("skipped", False))
    failed = len(results) - successful

    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    if skipped > 0:
        print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    # Group by optimizer
    print("\nResults by optimizer:")
    for optimizer in args.optimizers:
        opt_results = [r for r in results if r["optimizer"] == optimizer]
        opt_success = sum(1 for r in opt_results if r["success"])
        print(f"  {optimizer}: {opt_success}/{len(opt_results)} successful")

    print("\nCheck wandb for detailed results and loss curves.")
    print("Expected outcome: SGD should struggle more with deeper/wider CNNs")
    print("compared to APTS_D, showing slower convergence or worse final loss.")


if __name__ == "__main__":
    main()
