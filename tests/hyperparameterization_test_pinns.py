#!/usr/bin/env python3
"""
Hyperparameterization Test: SGD vs APTS_D for PINNs

This script tests the hypothesis that SGD performs worse than APTS_D on
overhyperparameterized Physics-Informed Neural Networks (PINNs). It runs both
optimizers on PINN problems with increasingly complex architectures to demonstrate
where SGD struggles.

Usage:
    python hyperparameterization_test_pinns.py
    python hyperparameterization_test_pinns.py --depths 4 8 16 32
    python hyperparameterization_test_pinns.py --widths 20 40 80 160
    python hyperparameterization_test_pinns.py --datasets poisson2d  # Default: 2D Poisson
    python hyperparameterization_test_pinns.py --datasets poisson1d --n-interior 1078 --n-boundary 2  # 1D with 1080 points
    python hyperparameterization_test_pinns.py --n-interior 1000 --n-boundary-side 27  # 2D with ~1108 points
    python hyperparameterization_test_pinns.py --initial-delta 0.05 --min-delta 0.0005 --max-delta 0.5  # Custom delta settings for APTS
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def create_config(
    optimizer: str,
    width: int,
    num_layers: int,
    base_dir: Path,
    dataset_name: str = "poisson1d",
    epochs: int = 5,
    batch_size: int = 1000,
    overlap: float = 0.0,
    batch_inc_factor: float = 1.0,
    max_wolfe_iters: int = None,
    max_zoom_iters: int = None,
    trial: int = 1,
    num_subdomains: int = None,
    num_replicas_per_subdomain: int = None,
    num_stages: int = None,
    n_interior: int = None,
    n_boundary: int = None,
    n_boundary_side: int = None,
    initial_delta: float = None,
    min_delta: float = None,
    max_delta: float = None,
) -> Path:
    """Create a configuration file for the given optimizer and PINN network size."""

    # Map dataset names to criterion names
    dataset_criterion_map = {
        "poisson1d": "pinn_poisson",
        "poisson2d": "pinn_poisson2d",
        "poisson3d": "pinn_poisson3d",
        "allencahn1d": "pinn_allencahn",
        "allencahn1d_time": "pinn_allencahn_time",
    }

    # Select base config template
    if optimizer == "sgd":
        base_config = base_dir / "config_files" / "config_sgd.yaml"
    else:
        # For APTS variants, use the PINN config as base
        base_config = base_dir / "config_files" / "config_apts_pinn.yaml"

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    # Load base config
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Update parameters common to all optimizers
    config["parameters"]["dataset_name"]["value"] = dataset_name
    config["parameters"]["model_name"]["value"] = "pinn_ffnn"
    config["parameters"]["criterion"]["value"] = dataset_criterion_map.get(
        dataset_name, "pinn_poisson"
    )
    config["parameters"]["batch_size"]["value"] = batch_size
    config["parameters"]["effective_batch_size"]["value"] = batch_size
    config["parameters"]["epochs"]["value"] = epochs
    config["parameters"]["seed"]["value"] = 42
    config["parameters"]["shuffle"]["value"] = False

    # Add width and num_layers parameters if they don't exist
    if "width" not in config["parameters"]:
        config["parameters"]["width"] = {}
    config["parameters"]["width"]["value"] = width

    if "num_layers" not in config["parameters"]:
        config["parameters"]["num_layers"] = {}
    config["parameters"]["num_layers"]["value"] = num_layers

    # Add trial parameter if it doesn't exist
    if "trial" not in config["parameters"]:
        config["parameters"]["trial"] = {}
    config["parameters"]["trial"]["value"] = trial

    # Optimizer-specific settings
    if optimizer == "sgd":
        config["parameters"]["optimizer"]["value"] = "sgd"
        config["parameters"]["learning_rate"]["value"] = 0.001  # Lower LR for PINNs
        config["parameters"]["num_subdomains"]["value"] = (
            num_subdomains if num_subdomains is not None else 1
        )
        config["parameters"]["overlap"]["value"] = overlap
        config["parameters"]["batch_inc_factor"]["value"] = batch_inc_factor
        config["parameters"]["max_iters"]["value"] = 0
    elif optimizer in ["apts_d", "apts_p"]:
        config["parameters"]["optimizer"]["value"] = optimizer
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
        if initial_delta is not None:
            if "delta" not in config["parameters"]:
                config["parameters"]["delta"] = {}
            config["parameters"]["delta"]["value"] = initial_delta
        if min_delta is not None:
            if "min_delta" not in config["parameters"]:
                config["parameters"]["min_delta"] = {}
            config["parameters"]["min_delta"]["value"] = min_delta
        if max_delta is not None:
            if "max_delta" not in config["parameters"]:
                config["parameters"]["max_delta"] = {}
            config["parameters"]["max_delta"]["value"] = max_delta
    elif optimizer == "apts_ip":
        config["parameters"]["optimizer"]["value"] = "apts_ip"
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
        if initial_delta is not None:
            if "delta" not in config["parameters"]:
                config["parameters"]["delta"] = {}
            config["parameters"]["delta"]["value"] = initial_delta
        if min_delta is not None:
            if "min_delta" not in config["parameters"]:
                config["parameters"]["min_delta"] = {}
            config["parameters"]["min_delta"]["value"] = min_delta
        if max_delta is not None:
            if "max_delta" not in config["parameters"]:
                config["parameters"]["max_delta"] = {}
            config["parameters"]["max_delta"]["value"] = max_delta
    elif optimizer == "apts_pinn":
        config["parameters"]["optimizer"]["value"] = "apts_pinn"
        # Keep existing APTS_PINN specific parameters from base config
        config["parameters"]["overlap"]["value"] = overlap
        config["parameters"]["batch_inc_factor"]["value"] = batch_inc_factor
        if num_subdomains is not None:
            config["parameters"]["num_subdomains"]["value"] = num_subdomains
        if max_wolfe_iters is not None:
            config["parameters"]["max_wolfe_iters"]["value"] = max_wolfe_iters
        if max_zoom_iters is not None:
            config["parameters"]["max_zoom_iters"]["value"] = max_zoom_iters
        if initial_delta is not None:
            if "delta" not in config["parameters"]:
                config["parameters"]["delta"] = {}
            config["parameters"]["delta"]["value"] = initial_delta
        if min_delta is not None:
            if "min_delta" not in config["parameters"]:
                config["parameters"]["min_delta"] = {}
            config["parameters"]["min_delta"]["value"] = min_delta
        if max_delta is not None:
            if "max_delta" not in config["parameters"]:
                config["parameters"]["max_delta"] = {}
            config["parameters"]["max_delta"]["value"] = max_delta

    # Set num_replicas_per_subdomain if provided (applies to all optimizers)
    if num_replicas_per_subdomain is not None:
        config["parameters"]["num_replicas_per_subdomain"][
            "value"
        ] = num_replicas_per_subdomain

    # Set dataset size parameters if provided
    if n_interior is not None:
        if "n_interior" not in config["parameters"]:
            config["parameters"]["n_interior"] = {}
        config["parameters"]["n_interior"]["value"] = n_interior

    if n_boundary is not None:
        if "n_boundary" not in config["parameters"]:
            config["parameters"]["n_boundary"] = {}
        config["parameters"]["n_boundary"]["value"] = n_boundary

    if n_boundary_side is not None:
        if "n_boundary_side" not in config["parameters"]:
            config["parameters"]["n_boundary_side"] = {}
        config["parameters"]["n_boundary_side"]["value"] = n_boundary_side

    # Create output config file
    config_name = f"config_hyperparam_pinn_{optimizer}_w{width}_nl{num_layers}_ds{dataset_name}_trial{trial}.yaml"
    output_path = base_dir / "config_files" / config_name

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def check_experiment_completed(config_path: Path) -> bool:
    """Check if an experiment has already been completed successfully."""
    # Look for wandb run directories that match this config
    job_name = config_path.stem
    wandb_dir = config_path.parent.parent / "wandb"

    if not wandb_dir.exists():
        return False

    # Check for successful run logs containing this job name
    for run_dir in wandb_dir.glob("*"):
        if run_dir.is_dir() and job_name in run_dir.name:
            # Check if run completed successfully
            debug_log = run_dir / "files" / "wandb-summary.json"
            if debug_log.exists():
                return True

    return False


def run_experiment(
    config_path: Path, test_dir: Path, skip_existing: bool = False
) -> Tuple[bool, str]:
    """Run a single experiment with the given config."""
    job_name = config_path.stem

    # Check if experiment already completed
    if skip_existing and check_experiment_completed(config_path):
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
            timeout=1200,  # 20 minute timeout (PINNs can take longer)
        )

        if result.returncode == 0:
            print(f"  ✓ Completed: {job_name}")
            return True, ""
        else:
            print(f"  ✗ Failed: {job_name}")
            return False, ""

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {job_name}")
        return False, "Experiment timed out"
    except Exception as e:
        print(f"  ✗ Error: {job_name}: {e}")
        return False, str(e)
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Test hyperparameterization effects on SGD vs APTS for PINNs"
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[20, 40],
        help="Network widths to test (default: 20 40)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Network depths (num_layers) to test (default: 4 8)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["poisson2d"],
        help="PINN datasets to test: poisson1d, poisson2d, poisson3d, allencahn1d, allencahn1d_time (default: poisson2d)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        type=str,
        default=["sgd", "apts_d"],
        help="Optimizers to test: sgd, apts_d, apts_p, apts_ip, apts_pinn (default: sgd apts_d)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=450,
        help="Number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)",
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
        "--n-interior",
        type=int,
        default=1000,
        help="Number of interior collocation points for PINN dataset (default: 1000 for 2D/3D, use 1078 for 1D to get 1080 total)",
    )
    parser.add_argument(
        "--n-boundary",
        type=int,
        default=None,
        help="Number of boundary points for 1D PINN dataset (default: 2 for poisson1d)",
    )
    parser.add_argument(
        "--n-boundary-side",
        type=int,
        default=27,
        help="Number of boundary points per side for 2D/3D PINN datasets (default: 27, which gives 108 boundary points for 2D [4 sides] or 162 for 3D [6 faces], totaling ~1080 points with n_interior=1000)",
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
    parser.add_argument(
        "--initial-delta",
        type=float,
        default=None,
        help="Initial trust region radius (delta) for APTS optimizers (optional, uses config default=0.1 if not set)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum trust region radius for APTS optimizers (optional, uses config default=0.001 if not set)",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=None,
        help="Maximum trust region radius for APTS optimizers (optional, uses config default=1.0 if not set)",
    )

    args = parser.parse_args()

    # Find test directory
    script_dir = Path(__file__).parent

    print("=" * 80)
    print("HYPERPARAMETERIZATION TEST: SGD vs APTS for PINNs")
    print("=" * 80)
    print(f"\nTest configuration:")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Widths: {args.widths}")
    print(f"  Depths: {args.depths}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Display dataset size information based on which parameters are set
    if args.n_boundary is not None:
        # 1D dataset with n_boundary
        print(f"  Dataset size: {args.n_interior} interior + {args.n_boundary} boundary = {args.n_interior + args.n_boundary} total points")
    elif args.n_boundary_side is not None and args.n_boundary_side > 0:
        # 2D/3D dataset with n_boundary_side
        boundary_2d = args.n_boundary_side * 4
        boundary_3d = args.n_boundary_side * 6
        print(f"  Dataset size: {args.n_interior} interior + {args.n_boundary_side} pts/side")
        print(f"    → 2D total: ~{args.n_interior + boundary_2d} points ({boundary_2d} boundary)")
        print(f"    → 3D total: ~{args.n_interior + boundary_3d} points ({boundary_3d} boundary)")
    else:
        print(f"  Dataset size: {args.n_interior} interior points")

    print(f"  Overlap: {args.overlap}")
    print(f"  Batch increase factor: {args.batch_inc_factor}")
    if args.max_wolfe_iters is not None:
        print(f"  Max Wolfe iters: {args.max_wolfe_iters}")
    if args.max_zoom_iters is not None:
        print(f"  Max zoom iters: {args.max_zoom_iters}")
    if args.initial_delta is not None:
        print(f"  Initial delta (APTS): {args.initial_delta}")
    if args.min_delta is not None:
        print(f"  Min delta (APTS): {args.min_delta}")
    if args.max_delta is not None:
        print(f"  Max delta (APTS): {args.max_delta}")
    print(f"  Number of trials: {args.num_trials}")
    if args.num_subdomains is not None:
        print(f"  Num subdomains: {args.num_subdomains}")
    if args.num_replicas_per_subdomain is not None:
        print(f"  Num replicas per subdomain: {args.num_replicas_per_subdomain}")
    if args.num_stages is not None:
        print(f"  Num stages: {args.num_stages}")
    print()

    # Clean old configs if requested
    if args.clean:
        print("Cleaning old PINN config files...")
        config_dir = script_dir / "config_files"
        for config_file in config_dir.glob("config_hyperparam_pinn_*.yaml"):
            config_file.unlink()
            print(f"  Removed: {config_file.name}")
        print()

    # Generate all configurations
    configs = []
    total_experiments = (
        len(args.optimizers)
        * len(args.datasets)
        * len(args.widths)
        * len(args.depths)
        * args.num_trials
    )

    print(f"Generating {total_experiments} configurations...")
    for optimizer in args.optimizers:
        for dataset in args.datasets:
            for width in args.widths:
                for depth in args.depths:
                    for trial in range(1, args.num_trials + 1):
                        try:
                            config_path = create_config(
                                optimizer=optimizer,
                                width=width,
                                num_layers=depth,
                                base_dir=script_dir,
                                dataset_name=dataset,
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
                                n_interior=args.n_interior,
                                n_boundary=args.n_boundary,
                                n_boundary_side=args.n_boundary_side,
                                initial_delta=args.initial_delta,
                                min_delta=args.min_delta,
                                max_delta=args.max_delta,
                            )
                            configs.append(
                                (optimizer, dataset, width, depth, trial, config_path)
                            )
                            print(f"  ✓ Created: {config_path.name}")
                        except Exception as e:
                            print(
                                f"  ✗ Failed to create config for {optimizer} dataset={dataset} w={width} d={depth} trial={trial}: {e}"
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
    for i, (optimizer, dataset, width, depth, trial, config_path) in enumerate(
        configs, 1
    ):
        print(
            f"[{i}/{len(configs)}] {optimizer} - Dataset: {dataset}, Width: {width}, Depth: {depth}, Trial: {trial}"
        )
        success, output = run_experiment(
            config_path, script_dir, skip_existing=args.skip_existing
        )
        results.append(
            {
                "optimizer": optimizer,
                "dataset": dataset,
                "width": width,
                "depth": depth,
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

    # Group by dataset
    print("\nResults by dataset:")
    for dataset in args.datasets:
        ds_results = [r for r in results if r["dataset"] == dataset]
        ds_success = sum(1 for r in ds_results if r["success"])
        print(f"  {dataset}: {ds_success}/{len(ds_results)} successful")

    print("\nCheck wandb for detailed results and loss curves.")
    print("Expected outcome: SGD should struggle more with deeper/wider PINN networks")
    print("compared to APTS, showing slower convergence or worse final loss.")


if __name__ == "__main__":
    main()
