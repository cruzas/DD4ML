#!/usr/bin/env python3
"""
Hyperparameterization Test: SGD vs APTS_D for GPT Models

This script tests the hypothesis that SGD performs worse than APTS_D on
overhyperparameterized networks. It runs both optimizers on TinyShakespeare with
increasingly complex GPT architectures to demonstrate where SGD struggles.

Usage:
    python hyperparameterization_test_gpt.py
    python hyperparameterization_test_gpt.py --n_layers 3 6 12
    python hyperparameterization_test_gpt.py --n_embds 48 96 192
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
    n_embd: int,
    n_head: int,
    n_layer: int,
    base_dir: Path,
    max_iters: int = 250,
    batch_size: int = 64,
    overlap: float = 0.0,
    batch_inc_factor: float = 1.0,
    max_wolfe_iters: int = None,
    max_zoom_iters: int = None,
    trial: int = 1,
    num_subdomains: int = None,
    num_replicas_per_subdomain: int = None,
    num_stages: int = None,
) -> Path:
    """Create a configuration file for the given optimizer and GPT architecture."""

    # Select base config template
    base_config = base_dir / "config_files" / f"config_{optimizer}.yaml"

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    # Load base config
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Add GPT-specific parameters if they don't exist
    if "n_embd" not in config["parameters"]:
        config["parameters"]["n_embd"] = {}
    if "n_head" not in config["parameters"]:
        config["parameters"]["n_head"] = {}
    if "n_layer" not in config["parameters"]:
        config["parameters"]["n_layer"] = {}
    if "model_type" not in config["parameters"]:
        config["parameters"]["model_type"] = {}

    # Add trial parameter if it doesn't exist
    if "trial" not in config["parameters"]:
        config["parameters"]["trial"] = {}

    # Update parameters for GPT on TinyShakespeare
    config["parameters"]["dataset_name"]["value"] = "tinyshakespeare"
    config["parameters"]["model_name"]["value"] = "nanogpt"
    # Set model_type to None when providing custom architecture parameters
    config["parameters"]["model_type"]["value"] = None
    config["parameters"]["n_embd"]["value"] = n_embd
    config["parameters"]["n_head"]["value"] = n_head
    config["parameters"]["n_layer"]["value"] = n_layer
    config["parameters"]["batch_size"]["value"] = batch_size
    config["parameters"]["effective_batch_size"]["value"] = batch_size
    config["parameters"]["epochs"]["value"] = 0
    config["parameters"]["max_iters"]["value"] = max_iters
    config["parameters"]["criterion"]["value"] = "cross_entropy_transformers"
    config["parameters"]["seed"]["value"] = 42
    config["parameters"]["shuffle"]["value"] = False
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
    config_name = f"config_hyperparam_gpt_{optimizer}_embd{n_embd}_head{n_head}_layer{n_layer}_trial{trial}.yaml"
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
            timeout=1200,  # 20 minute timeout for GPT models
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


def calculate_gpt_params(
    n_layer: int, n_head: int, n_embd: int, vocab_size: int = 65, block_size: int = 256
) -> int:
    """
    Calculate approximate number of parameters for a GPT model.

    Parameters:
    - n_layer: number of transformer blocks
    - n_head: number of attention heads
    - n_embd: embedding dimension
    - vocab_size: vocabulary size (default 65 for tinyshakespeare)
    - block_size: context length (default 256)
    """
    # Token embedding: vocab_size * n_embd
    # Position embedding: block_size * n_embd
    embedding_params = vocab_size * n_embd + block_size * n_embd

    # Per transformer block:
    # - LayerNorm 1: 2 * n_embd
    # - Attention (QKV projection): n_embd * 3 * n_embd
    # - Attention output projection: n_embd * n_embd
    # - LayerNorm 2: 2 * n_embd
    # - MLP fc: n_embd * 4 * n_embd
    # - MLP proj: 4 * n_embd * n_embd
    params_per_block = (
        2 * n_embd  # ln_1
        + n_embd * 3 * n_embd  # c_attn
        + n_embd * n_embd  # attn c_proj
        + 2 * n_embd  # ln_2
        + n_embd * 4 * n_embd  # mlp c_fc
        + 4 * n_embd * n_embd  # mlp c_proj
    )

    # Final LayerNorm: 2 * n_embd
    # Output head: n_embd * vocab_size
    output_params = 2 * n_embd + n_embd * vocab_size

    total = embedding_params + n_layer * params_per_block + output_params
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Test hyperparameterization effects on SGD vs APTS_D for GPT models"
    )
    parser.add_argument(
        "--n_embds",
        nargs="+",
        type=int,
        default=[48],
        help="Embedding dimensions to test (default: 48 96)",
    )
    parser.add_argument(
        "--n_heads",
        nargs="+",
        type=int,
        default=None,
        help="Number of attention heads to test (default: auto-calculated to divide n_embd evenly)",
    )
    parser.add_argument(
        "--n_layers",
        nargs="+",
        type=int,
        default=[3, 6],
        help="Number of transformer layers to test (default: 3 6)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        type=str,
        default=["sgd", "apts_d"],
        help="Optimizers to test: sgd, apts_d, apts_p, apts_ip (default: sgd apts_d)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=250,
        help="Maximum number of iterations to train (default: 250)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
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

    print("=" * 80)
    print("HYPERPARAMETERIZATION TEST: SGD vs APTS_D for GPT Models")
    print("=" * 80)
    print(f"\nTest configuration:")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Dataset: tinyshakespeare")
    print(f"  Criterion: cross_entropy_transformers")
    print(f"  Embedding dimensions: {args.n_embds}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Max iterations: {args.max_iters}")
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
    print()

    # Clean old configs if requested
    if args.clean:
        print("Cleaning old config files...")
        config_dir = script_dir / "config_files"
        for config_file in config_dir.glob("config_hyperparam_gpt_*.yaml"):
            config_file.unlink()
            print(f"  Removed: {config_file.name}")
        print()

    # Generate all configurations
    configs = []

    # Build configuration space
    for n_embd in args.n_embds:
        # Auto-calculate n_heads if not provided
        if args.n_heads is None:
            # Choose divisors of n_embd that make sense for attention heads
            # Common choices: n_embd/16, n_embd/8, etc.
            if n_embd >= 48:
                n_heads_list = [max(1, n_embd // 16), max(1, n_embd // 8)]
            else:
                n_heads_list = [max(1, n_embd // 8)]
            # Remove duplicates and ensure they divide n_embd evenly
            n_heads_list = sorted(
                list(set([h for h in n_heads_list if n_embd % h == 0]))
            )
        else:
            n_heads_list = [h for h in args.n_heads if n_embd % h == 0]

        if not n_heads_list:
            print(f"Warning: No valid n_head values for n_embd={n_embd}, skipping...")
            continue

        for n_head in n_heads_list:
            for n_layer in args.n_layers:
                # Calculate and display parameter count
                n_params = calculate_gpt_params(n_layer, n_head, n_embd)
                print(
                    f"Config: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer} → ~{n_params:,} params"
                )

                for optimizer in args.optimizers:
                    for trial in range(1, args.num_trials + 1):
                        try:
                            config_path = create_config(
                                optimizer=optimizer,
                                n_embd=n_embd,
                                n_head=n_head,
                                n_layer=n_layer,
                                base_dir=script_dir,
                                max_iters=args.max_iters,
                                batch_size=args.batch_size,
                                overlap=args.overlap,
                                batch_inc_factor=args.batch_inc_factor,
                                max_wolfe_iters=args.max_wolfe_iters,
                                max_zoom_iters=args.max_zoom_iters,
                                trial=trial,
                                num_subdomains=args.num_subdomains,
                                num_replicas_per_subdomain=args.num_replicas_per_subdomain,
                                num_stages=args.num_stages,
                            )
                            configs.append(
                                (optimizer, n_embd, n_head, n_layer, trial, config_path)
                            )
                            print(f"  ✓ Created: {config_path.name}")
                        except Exception as e:
                            print(
                                f"  ✗ Failed to create config for {optimizer} embd={n_embd} head={n_head} layer={n_layer} trial={trial}: {e}"
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
    for i, (optimizer, n_embd, n_head, n_layer, trial, config_path) in enumerate(
        configs, 1
    ):
        n_params = calculate_gpt_params(n_layer, n_head, n_embd)
        print(
            f"[{i}/{len(configs)}] {optimizer} - Embd: {n_embd}, Head: {n_head}, Layer: {n_layer}, Trial: {trial} (~{n_params:,} params)"
        )
        success, output = run_experiment(
            config_path, script_dir, skip_existing=args.skip_existing
        )
        results.append(
            {
                "optimizer": optimizer,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "trial": trial,
                "n_params": n_params,
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
    print("Expected outcome: SGD should struggle more with larger GPT models")
    print("compared to APTS_D, showing slower convergence or worse final loss.")


if __name__ == "__main__":
    main()
