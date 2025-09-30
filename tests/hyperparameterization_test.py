#!/usr/bin/env python3
"""
Hyperparameterization Test: SGD vs APTS_D

This script tests the hypothesis that SGD performs worse than APTS_D on
overhyperparameterized networks. It runs both optimizers on MNIST with
increasingly complex FFNN architectures to demonstrate where SGD struggles.

Usage:
    python hyperparameterization_test.py
    python hyperparameterization_test.py --depths 4 8 16 32
    python hyperparameterization_test.py --widths 128 256 512 1024
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
    epochs: int = 10,
    batch_size: int = 10000,
) -> Path:
    """Create a configuration file for the given optimizer and network size."""

    # Select base config template
    if optimizer == "sgd":
        base_config = base_dir / "config_files" / "config_sgd.yaml"
    elif optimizer == "apts_d":
        base_config = base_dir / "config_files" / "config_apts_d.yaml"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    # Load base config
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Update parameters
    config["parameters"]["dataset_name"]["value"] = "mnist"
    config["parameters"]["model_name"]["value"] = "medium_ffnn"
    config["parameters"]["width"]["value"] = width
    config["parameters"]["num_layers"]["value"] = num_layers
    config["parameters"]["batch_size"]["value"] = batch_size
    config["parameters"]["effective_batch_size"]["value"] = batch_size
    config["parameters"]["epochs"]["value"] = epochs
    config["parameters"]["max_iters"]["value"] = 0
    config["parameters"]["criterion"]["value"] = "cross_entropy"
    config["parameters"]["seed"]["value"] = 42
    config["parameters"]["shuffle"]["value"] = False

    # Optimizer-specific settings
    if optimizer == "sgd":
        config["parameters"]["learning_rate"]["value"] = 0.01
        config["parameters"]["num_subdomains"]["value"] = 1
    elif optimizer == "apts_d":
        config["parameters"]["num_subdomains"]["value"] = 2
        config["parameters"]["max_loc_iters"]["value"] = 2
        config["parameters"]["glob_second_order"]["value"] = False
        config["parameters"]["loc_second_order"]["value"] = False

    # Create output config file
    config_name = f"config_hyperparam_{optimizer}_w{width}_nl{num_layers}.yaml"
    output_path = base_dir / "config_files" / config_name

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def run_experiment(config_path: Path, test_dir: Path) -> Tuple[bool, str]:
    """Run a single experiment with the given config."""
    job_name = config_path.stem

    print(f"  Running: {job_name}")

    try:
        # Save current directory
        original_cwd = os.getcwd()

        # Change to test directory
        os.chdir(test_dir)

        # Run the experiment
        result = subprocess.run(
            [sys.executable, "run_config_file.py", "--sweep_config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"  ✓ Completed: {job_name}")
            return True, result.stdout
        else:
            print(f"  ✗ Failed: {job_name}")
            print(f"    Error: {result.stderr[:200]}")
            return False, result.stderr

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
        description="Test hyperparameterization effects on SGD vs APTS_D"
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Network widths to test (default: 128 256 512)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Network depths (num_layers) to test (default: 4 8 16)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        type=str,
        default=["sgd", "apts_d"],
        help="Optimizers to test (default: sgd apts_d)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size (default: 10000)",
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

    args = parser.parse_args()

    # Find test directory
    script_dir = Path(__file__).parent

    print("="*80)
    print("HYPERPARAMETERIZATION TEST: SGD vs APTS_D")
    print("="*80)
    print(f"\nTest configuration:")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Widths: {args.widths}")
    print(f"  Depths: {args.depths}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Clean old configs if requested
    if args.clean:
        print("Cleaning old config files...")
        config_dir = script_dir / "config_files"
        for config_file in config_dir.glob("config_hyperparam_*.yaml"):
            config_file.unlink()
            print(f"  Removed: {config_file.name}")
        print()

    # Generate all configurations
    configs = []
    total_experiments = len(args.optimizers) * len(args.widths) * len(args.depths)

    print(f"Generating {total_experiments} configurations...")
    for optimizer in args.optimizers:
        for width in args.widths:
            for depth in args.depths:
                try:
                    config_path = create_config(
                        optimizer=optimizer,
                        width=width,
                        num_layers=depth,
                        base_dir=script_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
                    configs.append((optimizer, width, depth, config_path))
                    print(f"  ✓ Created: {config_path.name}")
                except Exception as e:
                    print(f"  ✗ Failed to create config for {optimizer} w={width} d={depth}: {e}")

    print(f"\nGenerated {len(configs)} configurations")

    if args.dry_run:
        print("\n[DRY RUN] - Not running experiments")
        return

    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80 + "\n")

    results = []
    for i, (optimizer, width, depth, config_path) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {optimizer} - Width: {width}, Depth: {depth}")
        success, output = run_experiment(config_path, script_dir)
        results.append({
            "optimizer": optimizer,
            "width": width,
            "depth": depth,
            "success": success,
        })
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Group by optimizer
    print("\nResults by optimizer:")
    for optimizer in args.optimizers:
        opt_results = [r for r in results if r["optimizer"] == optimizer]
        opt_success = sum(1 for r in opt_results if r["success"])
        print(f"  {optimizer}: {opt_success}/{len(opt_results)} successful")

    print("\nCheck wandb for detailed results and loss curves.")
    print("Expected outcome: SGD should struggle more with deeper/wider networks")
    print("compared to APTS_D, showing slower convergence or worse final loss.")


if __name__ == "__main__":
    main()