#!/usr/bin/env python3
"""
Analysis Script for Hyperparameterization Tests - GPT Models

This script analyzes results from hyperparameterization_test_gpt.py experiments
stored in wandb under the 'ohtests' project. It compares SGD vs APTS_D (and
other optimizers) across different GPT architectures (n_embd, n_head, n_layer).

Usage:
    python analyze_hyperparam_results_gpt.py
    python analyze_hyperparam_results_gpt.py --entity your-wandb-username
    python analyze_hyperparam_results_gpt.py --save-plots
    python analyze_hyperparam_results_gpt.py --metric loss
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Enable LaTeX rendering and set font sizes
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 22,
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
    }
)

try:
    import wandb
except ImportError:
    print("Error: wandb package not found. Install with: pip install wandb")
    sys.exit(1)


def _get_cache_key(project_path: str, filters: Optional[Dict]) -> str:
    """Generate a stable cache key for project and filters."""
    canonical = json.dumps(
        {"project": project_path, "filters": filters or {}}, sort_keys=True, default=str
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _get_cache_dir() -> Path:
    """Get cache directory for storing wandb data."""
    cache_dir = Path.home() / ".cache" / "dd4ml_hyperparam_analysis_gpt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_runs(
    project: str = "ohtests",
    entity: Optional[str] = None,
    filters: Optional[Dict] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> List:
    """Fetch all runs from wandb project with caching."""
    api = wandb.Api()

    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project

    # Try cache first
    if use_cache:
        cache_dir = _get_cache_dir()
        cache_key = _get_cache_key(project_path, filters)
        cache_file = cache_dir / f"runs_{cache_key}.pkl"

        if cache_file.exists():
            try:
                cache_data = pd.read_pickle(cache_file)
                if isinstance(cache_data, dict) and "_fetched_at" in cache_data:
                    age_hours = (time.time() - cache_data["_fetched_at"]) / 3600
                    if age_hours <= cache_max_age_hours:
                        print(f"Using cached runs (age: {age_hours:.1f}h)")
                        return cache_data["runs"]
            except Exception:
                pass  # Fall through to fetch

    print(f"Fetching runs from wandb project: {project_path}")

    try:
        runs = api.runs(project_path, filters=filters)
        runs_list = list(runs)

        # Cache the results
        if use_cache:
            try:
                cache_data = {"_fetched_at": time.time(), "runs": runs_list}
                pd.to_pickle(cache_data, cache_file)
                print(f"Cached {len(runs_list)} runs")
            except Exception as e:
                print(f"Warning: Failed to cache runs: {e}")

        return runs_list
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print(
            f"\nTip: If the project is not found, specify your wandb entity with --entity"
        )
        sys.exit(1)


def format_optimizer_name(optimizer: str, num_subdomains: Optional[int] = None) -> str:
    """Format optimizer name for display with LaTeX.

    Args:
        optimizer: The optimizer name (e.g., 'sgd', 'apts_d', 'apts_p')
        num_subdomains: Number of subdomains (only relevant for APTS variants)
    """
    formatter = {
        "sgd": r"SGD",
        "apts_d": r"$\mathrm{SAPTS}_D$",
        "apts_p": r"$\mathrm{SAPTS}_P$",
        "apts_ip": r"$\mathrm{SAPTS}_{IP}$",
        "apts": r"$\mathrm{SAPTS}$",
    }
    base_name = formatter.get(optimizer.lower(), optimizer)

    # Add subdomain count for APTS variants if specified
    if num_subdomains is not None and optimizer.lower() in [
        "apts_d",
        "apts_p",
        "apts_ip",
    ]:
        base_name += f" ($N={num_subdomains}$)"

    return base_name


def _load_history_cached(run_id: str, run) -> pd.DataFrame:
    """Load run history from cache if available, otherwise fetch and cache."""
    cache_dir = _get_cache_dir() / "histories"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{run_id}.pkl"

    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass  # Fall through to fetch

    # Fetch from wandb
    history = run.history(samples=10000)

    # Cache it
    try:
        history.to_pickle(cache_file)
    except Exception:
        pass  # Continue even if caching fails

    return history


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


def validate_apts_parameters(df: pd.DataFrame) -> None:
    """Validate that APTS variants have overlap=0.33 and batch_inc_factor=1.5."""
    print("\n" + "=" * 80)
    print("VALIDATING APTS PARAMETERS")
    print("=" * 80)

    apts_variants = ["apts_d", "apts_p", "apts_ip"]
    issues_found = False

    for optimizer in apts_variants:
        opt_df = df[df["optimizer"] == optimizer]
        if len(opt_df) == 0:
            continue

        # Check overlap
        overlap_values = opt_df["overlap"].dropna().unique()
        if len(overlap_values) > 0:
            if not all(abs(v - 0.33) < 0.01 for v in overlap_values):
                print(
                    f"\n⚠ WARNING: {optimizer.upper()} has non-standard overlap values: {overlap_values}"
                )
                print(f"  Expected: 0.33")
                issues_found = True
            else:
                print(
                    f"\n✓ {optimizer.upper()}: overlap = {overlap_values[0]:.2f} (correct)"
                )

        # Check batch_inc_factor
        batch_inc_values = opt_df["batch_inc_factor"].dropna().unique()
        if len(batch_inc_values) > 0:
            if not all(abs(v - 1.5) < 0.01 for v in batch_inc_values):
                print(
                    f"⚠ WARNING: {optimizer.upper()} has non-standard batch_inc_factor values: {batch_inc_values}"
                )
                print(f"  Expected: 1.5")
                issues_found = True
            else:
                print(
                    f"✓ {optimizer.upper()}: batch_inc_factor = {batch_inc_values[0]:.2f} (correct)"
                )

    if not issues_found:
        print("\n✓ All APTS variants have correct parameters")


def extract_run_data(runs: List) -> pd.DataFrame:
    """Extract relevant data from wandb runs into a DataFrame."""
    results = []

    print(f"\nProcessing {len(runs)} runs...")

    for run in runs:
        config = run.config
        summary = run.summary._json_dict

        # Extract GPT architecture info
        n_embd = config.get("n_embd", None)
        n_head = config.get("n_head", None)
        n_layer = config.get("n_layer", None)

        # Convert to int if they are not None and not already int
        try:
            n_embd = int(n_embd) if n_embd is not None else None
        except (ValueError, TypeError):
            n_embd = None

        try:
            n_head = int(n_head) if n_head is not None else None
        except (ValueError, TypeError):
            n_head = None

        try:
            n_layer = int(n_layer) if n_layer is not None else None
        except (ValueError, TypeError):
            n_layer = None

        # Calculate total parameters (for GPT)
        if n_embd and n_head and n_layer:
            total_params = calculate_gpt_params(n_layer, n_head, n_embd)
        else:
            total_params = None

        # Extract optimizer info
        optimizer = config.get("optimizer", "unknown")

        # Extract trial number
        trial = config.get("trial", 1)

        # Extract training hyperparameters
        max_iters = config.get("max_iters", None)
        batch_size = config.get("batch_size", None)
        learning_rate = config.get("learning_rate", None)
        num_subdomains = config.get("num_subdomains", None)
        overlap = config.get("overlap", None)
        batch_inc_factor = config.get("batch_inc_factor", None)

        # Extract final metrics
        final_loss = summary.get("loss", None)
        final_accuracy = summary.get("accuracy", None)
        total_runtime = summary.get("running_time", None)
        total_grad_evals = summary.get("grad_evals", None)

        # Get history for convergence analysis (with caching)
        history = _load_history_cached(run.id, run)

        results.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "optimizer": optimizer,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "total_params": total_params,
                "trial": trial,
                "max_iters": max_iters,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_subdomains": num_subdomains,
                "overlap": overlap,
                "batch_inc_factor": batch_inc_factor,
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "total_runtime": total_runtime,
                "total_grad_evals": total_grad_evals,
                "history": history,
                "run_url": run.url,
            }
        )

    df = pd.DataFrame(results)

    # Filter out runs missing critical info
    initial_count = len(df)
    df = df.dropna(subset=["optimizer", "n_embd", "n_layer"])
    filtered_count = initial_count - len(df)

    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} runs missing critical config info")

    print(f"  Successfully processed {len(df)} runs")

    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics of the experiments."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal runs: {len(df)}")
    print(f"Optimizers tested: {sorted(df['optimizer'].unique())}")
    print(f"Embedding dimensions: {sorted(df['n_embd'].unique())}")
    print(f"Number of attention heads: {sorted(df['n_head'].unique())}")
    print(f"Number of layers: {sorted(df['n_layer'].unique())}")
    print(f"Trials per configuration: {df['trial'].max()}")

    # Group by optimizer
    print("\n" + "-" * 80)
    print("Results by Optimizer")
    print("-" * 80)

    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]
        print(f"\n{optimizer.upper()}:")
        print(f"  Runs: {len(opt_df)}")
        print(
            f"  Avg final loss: {opt_df['final_loss'].mean():.6f} ± {opt_df['final_loss'].std():.6f}"
        )
        if opt_df["final_accuracy"].notna().any():
            print(
                f"  Avg final accuracy: {opt_df['final_accuracy'].mean():.4f} ± {opt_df['final_accuracy'].std():.4f}"
            )
        if opt_df["total_runtime"].notna().any():
            print(
                f"  Avg runtime: {opt_df['total_runtime'].mean():.2f}s ± {opt_df['total_runtime'].std():.2f}s"
            )
        if opt_df["total_grad_evals"].notna().any():
            print(f"  Avg grad evals: {opt_df['total_grad_evals'].mean():.0f}")


def print_comparison_by_architecture(df: pd.DataFrame) -> None:
    """Compare optimizers across different GPT architectures."""
    print("\n" + "=" * 80)
    print("COMPARISON BY ARCHITECTURE")
    print("=" * 80)

    # Group by network architecture
    architectures = (
        df.groupby(["n_embd", "n_head", "n_layer"])
        .size()
        .reset_index()[["n_embd", "n_head", "n_layer"]]
    )

    for _, arch in architectures.iterrows():
        n_embd = arch["n_embd"]
        n_head = arch["n_head"]
        n_layer = arch["n_layer"]

        arch_df = df[
            (df["n_embd"] == n_embd)
            & (df["n_head"] == n_head)
            & (df["n_layer"] == n_layer)
        ]

        if len(arch_df) == 0:
            continue

        print(f"\n{'─' * 80}")
        print(
            f"Embd={n_embd}, Head={n_head}, Layer={n_layer} ({arch_df['total_params'].iloc[0]:,} parameters)"
        )
        print(f"{'─' * 80}")

        # Compare optimizers for this architecture
        comparison = (
            arch_df.groupby("optimizer")
            .agg(
                {
                    "final_loss": ["mean", "std", "min"],
                    "final_accuracy": ["mean", "std", "max"],
                    "total_runtime": ["mean", "std"],
                    "total_grad_evals": ["mean", "std"],
                }
            )
            .round(6)
        )

        print(comparison.to_string())


def print_hyperparameterization_analysis(df: pd.DataFrame) -> None:
    """Analyze how model size affects each optimizer."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETERIZATION EFFECT ANALYSIS")
    print("=" * 80)
    print("\nHow does increasing model size affect each optimizer?\n")

    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]

        print(f"\n{optimizer.upper()}:")
        print(f"{'─' * 60}")

        # Sort by total parameters
        summary = (
            opt_df.groupby(["n_embd", "n_head", "n_layer", "total_params"])
            .agg(
                {
                    "final_loss": ["mean", "std"],
                    "final_accuracy": ["mean", "std"],
                }
            )
            .reset_index()
            .sort_values("total_params")
        )

        print(
            f"{'Params':<12} {'Embd':<8} {'Head':<8} {'Layer':<8} {'Loss (mean±std)':<20} {'Accuracy (mean±std)':<20}"
        )
        print("─" * 80)

        for _, row in summary.iterrows():
            params = f"{int(row['total_params']):,}"
            embd = int(row["n_embd"])
            head = int(row["n_head"])
            layer = int(row["n_layer"])
            loss_mean = row[("final_loss", "mean")]
            loss_std = row[("final_loss", "std")]

            # Handle potentially missing accuracy data
            if pd.notna(row[("final_accuracy", "mean")]):
                acc_mean = row[("final_accuracy", "mean")]
                acc_std = row[("final_accuracy", "std")]
                acc_str = f"{acc_mean:.4f}±{acc_std:.4f}"
            else:
                acc_str = "N/A"

            print(
                f"{params:<12} {embd:<8} {head:<8} {layer:<8} {loss_mean:.6f}±{loss_std:.6f}    {acc_str}"
            )


def create_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None, sgd_overlap: float = 0.0
) -> None:
    """Create comparison plots across network sizes.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots (if None, shows plots instead)
        sgd_overlap: Overlap value to use for SGD data (default: 0.0 for no overlap)
    """
    print("" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    print(f"Using SGD with overlap={sgd_overlap:.2f}")

    # Removed old docstring continuation
    print("" + "=" * 80)
    print(
        "GENERATING COMPARISON PLOTS across model sizes, separated by parameter combinations."
        ""
    )
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    sns.set_style("whitegrid")

    # Identify parameter combinations from all optimizers
    param_combinations = []

    # Get unique combinations of overlap and batch_inc_factor across all optimizers
    all_params = df[["overlap", "batch_inc_factor"]].drop_duplicates()
    for _, row in all_params.iterrows():
        overlap = row["overlap"]
        batch_inc = row["batch_inc_factor"]
        if pd.notna(overlap) and pd.notna(batch_inc):
            param_combinations.append((overlap, batch_inc))

    if param_combinations:
        print(f"\nFound {len(param_combinations)} parameter combination(s):")
        for overlap, batch_inc in param_combinations:
            # Count which optimizers have this combination
            matching_runs = df[
                (abs(df["overlap"] - overlap) < 0.01)
                & (abs(df["batch_inc_factor"] - batch_inc) < 0.01)
            ]
            optimizers = sorted(matching_runs["optimizer"].unique())
            print(
                f"  - overlap={overlap:.2f}, batch_inc_factor={batch_inc:.2f}: {optimizers}"
            )

    # If no parameter combinations found or all are NaN, use default grouping
    if not param_combinations:
        print("\nNo parameter information found, creating combined plots...")
        param_combinations = [(None, None)]

    # Create plots for each parameter combination
    for overlap_val, batch_inc_val in param_combinations:
        if overlap_val is not None and batch_inc_val is not None:
            print(
                f"\n--- Creating plots for overlap={overlap_val:.2f}, batch_inc_factor={batch_inc_val:.2f} ---"
            )

            # Filter APTS variants to this parameter combination
            # But always use SGD with no overlap (best performing configuration)
            apts_subset = df[
                (df["optimizer"].isin(["apts_d", "apts_p", "apts_ip"]))
                & (abs(df["overlap"] - overlap_val) < 0.01)
                & (abs(df["batch_inc_factor"] - batch_inc_val) < 0.01)
            ]

            # Use SGD with specified overlap value
            sgd_subset = df[
                (df["optimizer"] == "sgd") & (abs(df["overlap"] - sgd_overlap) < 0.01)
            ]

            # Combine SGD (no overlap) with APTS (with specified overlap/batch_inc)
            df_subset = pd.concat([sgd_subset, apts_subset])

            suffix = f"_overlap{overlap_val:.2f}_batchinc{batch_inc_val:.2f}".replace(
                ".", "_"
            )
        else:
            print("\n--- Creating combined plots (no parameter filtering) ---")
            df_subset = df
            suffix = ""

        if len(df_subset) == 0:
            print(f"  No data for this combination, skipping...")
            continue

        # Aggregate across trials and architectures
        # For APTS variants, also group by num_subdomains
        groupby_cols = ["optimizer", "total_params"]

        # Add num_subdomains to grouping for APTS variants
        apts_variants = ["apts_d", "apts_p", "apts_ip"]
        if any(opt in df_subset["optimizer"].unique() for opt in apts_variants):
            groupby_cols.append("num_subdomains")

        agg_df = (
            df_subset.groupby(groupby_cols)
            .agg(
                {
                    "final_loss": ["mean", "std"],
                    "final_accuracy": ["mean", "std"],
                    "total_runtime": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in agg_df.columns.values
        ]

        # Plot 1: Final Loss vs Model Size
        fig, ax = plt.subplots(figsize=(10, 6))

        apts_variants = ["apts_d", "apts_p", "apts_ip"]
        for optimizer in sorted(agg_df["optimizer"].unique()):
            # For APTS variants, plot each subdomain count separately
            if optimizer in apts_variants and "num_subdomains" in agg_df.columns:
                for num_subs in sorted(
                    agg_df[agg_df["optimizer"] == optimizer]["num_subdomains"]
                    .dropna()
                    .unique()
                ):
                    opt_df = agg_df[
                        (agg_df["optimizer"] == optimizer)
                        & (agg_df["num_subdomains"] == num_subs)
                    ].sort_values("total_params")
                    ax.errorbar(
                        opt_df["total_params"],
                        opt_df["final_loss_mean"],
                        yerr=opt_df["final_loss_std"],
                        marker="o",
                        capsize=5,
                        label=format_optimizer_name(optimizer, int(num_subs)),
                        linewidth=2,
                        markersize=8,
                    )
            else:
                # For SGD and other optimizers, plot normally
                opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values(
                    "total_params"
                )
                ax.errorbar(
                    opt_df["total_params"],
                    opt_df["final_loss_mean"],
                    yerr=opt_df["final_loss_std"],
                    marker="o",
                    capsize=5,
                    label=format_optimizer_name(optimizer),
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel(r"Number of parameters", fontsize=12)
        ax.set_ylabel(r"Final avg. loss", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        # Add parameter values as text annotation if they exist
        if overlap_val is not None and batch_inc_val is not None:
            param_text = (
                f"SAPTS: ov={overlap_val:.2f}, bif={batch_inc_val:.2f}\n"
                f"SGD: ov={sgd_overlap:.2f}"
            )
            ax.text(
                0.98,
                0.98,
                param_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"loss_vs_model_size_gpt{suffix}.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

        # Plot 2: Final Accuracy vs Model Size (if accuracy data exists)
        if agg_df["final_accuracy_mean"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 6))

            apts_variants = ["apts_d", "apts_p", "apts_ip"]
            for optimizer in sorted(agg_df["optimizer"].unique()):
                # For APTS variants, plot each subdomain count separately
                if optimizer in apts_variants and "num_subdomains" in agg_df.columns:
                    for num_subs in sorted(
                        agg_df[agg_df["optimizer"] == optimizer]["num_subdomains"]
                        .dropna()
                        .unique()
                    ):
                        opt_df = agg_df[
                            (agg_df["optimizer"] == optimizer)
                            & (agg_df["num_subdomains"] == num_subs)
                        ].sort_values("total_params")
                        # Filter out NaN accuracy values
                        opt_df = opt_df.dropna(subset=["final_accuracy_mean"])
                        if len(opt_df) > 0:
                            ax.errorbar(
                                opt_df["total_params"],
                                opt_df["final_accuracy_mean"],
                                yerr=opt_df["final_accuracy_std"],
                                marker="o",
                                capsize=5,
                                label=format_optimizer_name(optimizer, int(num_subs)),
                                linewidth=2,
                                markersize=8,
                            )
                else:
                    # For SGD and other optimizers, plot normally
                    opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values(
                        "total_params"
                    )
                    # Filter out NaN accuracy values
                    opt_df = opt_df.dropna(subset=["final_accuracy_mean"])
                    if len(opt_df) > 0:
                        ax.errorbar(
                            opt_df["total_params"],
                            opt_df["final_accuracy_mean"],
                            yerr=opt_df["final_accuracy_std"],
                            marker="o",
                            capsize=5,
                            label=format_optimizer_name(optimizer),
                            linewidth=2,
                            markersize=8,
                        )

            ax.set_xlabel(r"Number of parameters", fontsize=12)
            ax.set_ylabel(r"Final avg. accuracy (\%)", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            # Add parameter values as text annotation if they exist
            if overlap_val is not None and batch_inc_val is not None:
                param_text = (
                    f"SAPTS: ov={overlap_val:.2f}, bif={batch_inc_val:.2f}\n"
                    f"SGD: ov={sgd_overlap:.2f}"
                )
                ax.text(
                    0.98,
                    0.02,
                    param_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            plt.tight_layout()

            if output_dir:
                filepath = output_dir / f"accuracy_vs_model_size_gpt{suffix}.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()

    # Plot 3: Heatmap of loss by embedding dimension and number of layers (parameter-aware)
    # (fixing n_head to most common value for simplicity)
    most_common_n_head = df["n_head"].mode()[0]

    # First, calculate global min/max across ALL data for consistent scaling
    print("\n" + "=" * 80)
    print("CALCULATING GLOBAL HEATMAP SCALES")
    print("=" * 80)
    print(f"Using SGD with overlap={sgd_overlap:.2f}")

    all_loss_pivots_global = []
    apts_variants = ["apts_d", "apts_p", "apts_ip"]

    # For SGD, use specified overlap value
    sgd_df = df[df["optimizer"] == "sgd"]
    sgd_selected = sgd_df[
        (sgd_df["overlap"].notna()) & (abs(sgd_df["overlap"] - sgd_overlap) < 0.01)
    ]
    if len(sgd_selected) > 0:
        pivot = sgd_selected[sgd_selected["n_head"] == most_common_n_head].pivot_table(
            values="final_loss", index="n_layer", columns="n_embd", aggfunc="mean"
        )
        if not pivot.empty:
            all_loss_pivots_global.append(pivot)

    # For APTS variants with overlap=0.33, batch_inc=1.5
    for optimizer in apts_variants:
        opt_df = df[
            (df["optimizer"] == optimizer)
            & (df["overlap"].notna())
            & (df["batch_inc_factor"].notna())
            & (abs(df["overlap"] - 0.33) < 0.01)
            & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
        ]
        for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
            sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
            sub_df = sub_df[sub_df["n_head"] == most_common_n_head]
            pivot = sub_df.pivot_table(
                values="final_loss",
                index="n_layer",
                columns="n_embd",
                aggfunc="mean",
            )
            if not pivot.empty:
                all_loss_pivots_global.append(pivot)

    if all_loss_pivots_global:
        global_loss_min = min(pivot.min().min() for pivot in all_loss_pivots_global)
        global_loss_max = max(pivot.max().max() for pivot in all_loss_pivots_global)
        print(
            f"\nGlobal loss heatmap scale: {global_loss_min:.4f} to {global_loss_max:.4f}"
        )
    else:
        global_loss_min = None
        global_loss_max = None

    all_acc_pivots_global = []

    # For SGD, use specified overlap value
    if len(sgd_selected) > 0:
        pivot = sgd_selected[sgd_selected["n_head"] == most_common_n_head].pivot_table(
            values="final_accuracy", index="n_layer", columns="n_embd", aggfunc="mean"
        )
        if not pivot.empty:
            all_acc_pivots_global.append(pivot)

    # For APTS variants with overlap=0.33, batch_inc=1.5
    for optimizer in apts_variants:
        opt_df = df[
            (df["optimizer"] == optimizer)
            & (df["overlap"].notna())
            & (df["batch_inc_factor"].notna())
            & (abs(df["overlap"] - 0.33) < 0.01)
            & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
        ]
        for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
            sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
            sub_df = sub_df[sub_df["n_head"] == most_common_n_head]
            pivot = sub_df.pivot_table(
                values="final_accuracy",
                index="n_layer",
                columns="n_embd",
                aggfunc="mean",
            )
            if not pivot.empty:
                all_acc_pivots_global.append(pivot)

    if all_acc_pivots_global:
        global_acc_min = min(pivot.min().min() for pivot in all_acc_pivots_global)
        global_acc_max = max(pivot.max().max() for pivot in all_acc_pivots_global)
        print(
            f"Global accuracy heatmap scale: {global_acc_min:.4f} to {global_acc_max:.4f}"
        )
    else:
        global_acc_min = None
        global_acc_max = None

    # Generate combined heatmaps (SGD with selected overlap + APTS with overlap=0.33, batch_inc=1.5)
    print("\n--- Creating combined heatmaps ---")

    # Collect all heatmap data for LOSS
    if global_loss_min is not None and global_loss_max is not None:
        heatmap_data = []

        # Add SGD with selected overlap
        if len(sgd_selected) > 0:
            pivot = sgd_selected[sgd_selected["n_head"] == most_common_n_head].pivot_table(
                values="final_loss",
                index="n_layer",
                columns="n_embd",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append({
                    "pivot": pivot,
                    "title": format_optimizer_name("sgd"),
                    "optimizer": "sgd",
                    "num_subs": None
                })

        # Add APTS variants with overlap=0.33, batch_inc=1.5
        for optimizer in apts_variants:
            opt_df = df[
                (df["optimizer"] == optimizer)
                & (df["overlap"].notna())
                & (df["batch_inc_factor"].notna())
                & (abs(df["overlap"] - 0.33) < 0.01)
                & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
                & (df["n_head"] == most_common_n_head)
            ]
            for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                pivot = sub_df.pivot_table(
                    values="final_loss",
                    index="n_layer",
                    columns="n_embd",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append({
                        "pivot": pivot,
                        "title": format_optimizer_name(optimizer, int(num_subs)),
                        "optimizer": optimizer,
                        "num_subs": int(num_subs)
                    })

        # Create combined figure with all loss heatmaps
        if heatmap_data:
            n_heatmaps = len(heatmap_data)
            n_cols = min(3, n_heatmaps)  # Max 3 columns
            n_rows = (n_heatmaps + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Flatten axes array for easier indexing
            if n_heatmaps == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1).flatten()
            else:
                axes = axes.flatten()

            # Plot each heatmap
            for idx, data in enumerate(heatmap_data):
                ax = axes[idx]
                row = idx // n_cols
                col = idx % n_cols

                sns.heatmap(
                    data["pivot"],
                    annot=True,
                    fmt=".4f",
                    cmap="RdYlGn_r",
                    ax=ax,
                    vmin=global_loss_min,
                    vmax=global_loss_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(col == 0),  # Show y tick labels only on leftmost column
                )
                ax.set_title(f"{data['title']} (n\_head={most_common_n_head})", fontsize=14, fontweight="bold")

                # Only show x-label on center plot of bottom row
                if row == n_rows - 1 and col == n_cols // 2:
                    ax.set_xlabel(r"Embedding dimension", fontsize=12)
                else:
                    ax.set_xlabel("")

                # Only show y-label on leftmost column
                if col == 0:
                    ax.set_ylabel(r"Number of layers", fontsize=12)
                else:
                    ax.set_ylabel("")

            # Hide unused subplots
            for idx in range(n_heatmaps, len(axes)):
                axes[idx].axis('off')

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=global_loss_min, vmax=global_loss_max)
            sm = cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. loss", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_loss_combined_gpt.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()

    # Collect all heatmap data for ACCURACY
    if global_acc_min is not None and global_acc_max is not None:
        heatmap_data = []

        # Add SGD with selected overlap
        if len(sgd_selected) > 0:
            pivot = sgd_selected[sgd_selected["n_head"] == most_common_n_head].pivot_table(
                values="final_accuracy",
                index="n_layer",
                columns="n_embd",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append({
                    "pivot": pivot,
                    "title": format_optimizer_name("sgd"),
                    "optimizer": "sgd",
                    "num_subs": None
                })

        # Add APTS variants with overlap=0.33, batch_inc=1.5
        for optimizer in apts_variants:
            opt_df = df[
                (df["optimizer"] == optimizer)
                & (df["overlap"].notna())
                & (df["batch_inc_factor"].notna())
                & (abs(df["overlap"] - 0.33) < 0.01)
                & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
                & (df["n_head"] == most_common_n_head)
            ]
            for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                pivot = sub_df.pivot_table(
                    values="final_accuracy",
                    index="n_layer",
                    columns="n_embd",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append({
                        "pivot": pivot,
                        "title": format_optimizer_name(optimizer, int(num_subs)),
                        "optimizer": optimizer,
                        "num_subs": int(num_subs)
                    })

        # Create combined figure with all accuracy heatmaps
        if heatmap_data:
            n_heatmaps = len(heatmap_data)
            n_cols = min(3, n_heatmaps)  # Max 3 columns
            n_rows = (n_heatmaps + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Flatten axes array for easier indexing
            if n_heatmaps == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1).flatten()
            else:
                axes = axes.flatten()

            # Plot each heatmap
            for idx, data in enumerate(heatmap_data):
                ax = axes[idx]
                row = idx // n_cols
                col = idx % n_cols

                sns.heatmap(
                    data["pivot"],
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    ax=ax,
                    vmin=global_acc_min,
                    vmax=global_acc_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(col == 0),  # Show y tick labels only on leftmost column
                )
                ax.set_title(f"{data['title']} (n\_head={most_common_n_head})", fontsize=14, fontweight="bold")

                # Only show x-label on center plot of bottom row
                if row == n_rows - 1 and col == n_cols // 2:
                    ax.set_xlabel(r"Embedding dimension", fontsize=12)
                else:
                    ax.set_xlabel("")

                # Only show y-label on leftmost column
                if col == 0:
                    ax.set_ylabel(r"Number of layers", fontsize=12)
                else:
                    ax.set_ylabel("")

            # Hide unused subplots
            for idx in range(n_heatmaps, len(axes)):
                axes[idx].axis('off')

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=global_acc_min, vmax=global_acc_max)
            sm = cm.ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. accuracy (\%)", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_accuracy_combined_gpt.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()


def create_sgd_parameter_comparison_plots_gpt(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create comparison plots for SGD with different parameter combinations."""
    print("\n" + "=" * 80)
    print("GENERATING SGD PARAMETER COMPARISON PLOTS")
    print("=" * 80)

    # Filter to SGD only
    sgd_df = df[df["optimizer"] == "sgd"].copy()

    if len(sgd_df) == 0:
        print("\nNo SGD data found, skipping comparison plots...")
        return

    # Identify unique parameter combinations for SGD
    sgd_params = sgd_df[["overlap", "batch_inc_factor"]].drop_duplicates()
    param_combinations = []

    for _, row in sgd_params.iterrows():
        overlap = row["overlap"]
        batch_inc = row["batch_inc_factor"]
        if pd.notna(overlap) and pd.notna(batch_inc):
            param_combinations.append((overlap, batch_inc))

    if len(param_combinations) < 2:
        print(f"\nFound only {len(param_combinations)} SGD parameter combination(s).")
        print("Need at least 2 combinations for comparison. Skipping...")
        return

    print(f"\nFound {len(param_combinations)} SGD parameter combination(s):")
    for overlap, batch_inc in param_combinations:
        print(f"  - overlap={overlap:.2f}, batch_inc_factor={batch_inc:.2f}")

    sns.set_style("whitegrid")

    # Plot 1: Side-by-side line plot (Loss vs Model Size)
    fig, ax = plt.subplots(figsize=(10, 6))

    for overlap_val, batch_inc_val in param_combinations:
        # Filter SGD to this parameter combination
        sgd_subset = sgd_df[
            (abs(sgd_df["overlap"] - overlap_val) < 0.01)
            & (abs(sgd_df["batch_inc_factor"] - batch_inc_val) < 0.01)
        ]

        if len(sgd_subset) == 0:
            continue

        # Aggregate across trials
        agg_df = (
            sgd_subset.groupby("total_params")
            .agg({"final_loss": ["mean", "std"]})
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = ["total_params", "loss_mean", "loss_std"]

        label = (
            f"overlap={overlap_val*100:.0f}\\%, batch inc. factor={batch_inc_val:.2f}"
        )
        ax.errorbar(
            agg_df["total_params"],
            agg_df["loss_mean"],
            yerr=agg_df["loss_std"],
            marker="o",
            capsize=5,
            label=label,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel(r"Number of parameters", fontsize=12)
    ax.set_ylabel(r"Final avg. loss (SGD)", fontsize=12)
    ax.set_title(
        r"SGD: Parameter Combination Comparison", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()

    if output_dir:
        filepath = output_dir / "sgd_parameter_comparison_loss_gpt.pdf"
        plt.savefig(filepath, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    else:
        plt.show()

    plt.close()

    # Plot 2: Side-by-side line plot (Accuracy vs Model Size)
    if sgd_df["final_accuracy"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))

        for overlap_val, batch_inc_val in param_combinations:
            # Filter SGD to this parameter combination
            sgd_subset = sgd_df[
                (abs(sgd_df["overlap"] - overlap_val) < 0.01)
                & (abs(sgd_df["batch_inc_factor"] - batch_inc_val) < 0.01)
            ]

            if len(sgd_subset) == 0:
                continue

            # Aggregate across trials
            agg_df = (
                sgd_subset.groupby("total_params")
                .agg({"final_accuracy": ["mean", "std"]})
                .reset_index()
            )

            # Flatten column names
            agg_df.columns = ["total_params", "accuracy_mean", "accuracy_std"]

            label = f"overlap={overlap_val*100:.0f}\\%, batch inc. factor={batch_inc_val:.2f}"
            ax.errorbar(
                agg_df["total_params"],
                agg_df["accuracy_mean"],
                yerr=agg_df["accuracy_std"],
                marker="o",
                capsize=5,
                label=label,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel(r"Number of parameters", fontsize=12)
        ax.set_ylabel(r"Final avg. accuracy (SGD) (\%)", fontsize=12)
        ax.set_title(
            r"SGD: Parameter Combination Comparison", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / "sgd_parameter_comparison_accuracy_gpt.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    # Plot 3 & 4: Scatter plots for pairwise comparison
    if len(param_combinations) == 2:
        overlap1, batch_inc1 = param_combinations[0]
        overlap2, batch_inc2 = param_combinations[1]

        # Get data for both combinations
        sgd1 = sgd_df[
            (abs(sgd_df["overlap"] - overlap1) < 0.01)
            & (abs(sgd_df["batch_inc_factor"] - batch_inc1) < 0.01)
        ].copy()
        sgd2 = sgd_df[
            (abs(sgd_df["overlap"] - overlap2) < 0.01)
            & (abs(sgd_df["batch_inc_factor"] - batch_inc2) < 0.01)
        ].copy()

        # Create labels for debugging
        label1 = f"overlap={overlap1*100:.0f}\\%, batch inc. factor={batch_inc1:.2f}"
        label2 = f"overlap={overlap2*100:.0f}\\%, batch inc. factor={batch_inc2:.2f}"

        print(f"\n  Found {len(sgd1)} runs for Config 1 ({label1})")
        print(f"  Found {len(sgd2)} runs for Config 2 ({label2})")

        # Aggregate by architecture (n_embd, n_head, n_layer)
        sgd1_agg = (
            sgd1.groupby(["n_embd", "n_head", "n_layer"])
            .agg(
                {
                    "final_loss": "mean",
                    "final_accuracy": "mean",
                    "total_params": "first",
                }
            )
            .reset_index()
        )
        sgd2_agg = (
            sgd2.groupby(["n_embd", "n_head", "n_layer"])
            .agg(
                {
                    "final_loss": "mean",
                    "final_accuracy": "mean",
                    "total_params": "first",
                }
            )
            .reset_index()
        )

        print(f"  Config 1: {len(sgd1_agg)} unique architectures")
        print(f"  Config 2: {len(sgd2_agg)} unique architectures")

        # Merge on architecture
        merged = pd.merge(
            sgd1_agg,
            sgd2_agg,
            on=["n_embd", "n_head", "n_layer", "total_params"],
            suffixes=("_1", "_2"),
        )

        print(f"  Architectures present in both configs: {len(merged)}")

        # Check for NaN values
        if len(merged) > 0:
            loss_valid = (
                merged[["final_loss_1", "final_loss_2"]].notna().all(axis=1).sum()
            )
            acc_valid = (
                merged[["final_accuracy_1", "final_accuracy_2"]]
                .notna()
                .all(axis=1)
                .sum()
            )
            print(f"  Architectures with valid loss data: {loss_valid}")
            print(f"  Architectures with valid accuracy data: {acc_valid}")

        if len(merged) > 0:
            # Scatter plot for loss
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.scatter(
                merged["final_loss_1"],
                merged["final_loss_2"],
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
            )

            # Add diagonal line
            min_val = min(merged["final_loss_1"].min(), merged["final_loss_2"].min())
            max_val = max(merged["final_loss_1"].max(), merged["final_loss_2"].max())
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1
            )

            ax.set_xlabel(f"Final loss: {label1}", fontsize=18)
            ax.set_ylabel(f"Final loss: {label2}", fontsize=18)
            ax.set_title(r"SGD: Direct Loss Comparison", fontsize=20, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Create descriptive labels
            def get_config_label(overlap, batch_inc):
                if abs(overlap) < 0.01 and abs(batch_inc - 1.0) < 0.01:
                    return "no overlap \\& no batch increase"
                elif abs(overlap - 0.33) < 0.01 and abs(batch_inc - 1.5) < 0.01:
                    return "overlap \\& batch increase"
                else:
                    return f"overlap={overlap*100:.0f}\\%, batch inc. factor={batch_inc:.2f}"

            config1_label = get_config_label(overlap1, batch_inc1)
            config2_label = get_config_label(overlap2, batch_inc2)

            # Add text indicating which is better
            ax.text(
                0.05,
                0.95,
                f"Below diagonal = {config2_label} better\nAbove diagonal = {config1_label} better",
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()

            if output_dir:
                filepath = output_dir / "sgd_parameter_scatter_loss_gpt.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()

            # Scatter plot for accuracy
            if (
                merged["final_accuracy_1"].notna().any()
                and merged["final_accuracy_2"].notna().any()
            ):
                fig, ax = plt.subplots(figsize=(8, 8))

                ax.scatter(
                    merged["final_accuracy_1"],
                    merged["final_accuracy_2"],
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=1.5,
                )

                # Add diagonal line
                min_val = min(
                    merged["final_accuracy_1"].min(), merged["final_accuracy_2"].min()
                )
                max_val = max(
                    merged["final_accuracy_1"].max(), merged["final_accuracy_2"].max()
                )
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    alpha=0.5,
                    linewidth=1,
                )

                ax.set_xlabel(f"Final accuracy (\\%): {label1}", fontsize=18)
                ax.set_ylabel(f"Final accuracy (\\%): {label2}", fontsize=18)
                ax.set_title(
                    r"SGD: Direct Accuracy Comparison", fontsize=20, fontweight="bold"
                )
                ax.grid(True, alpha=0.3)

                # Add text indicating which is better
                ax.text(
                    0.05,
                    0.95,
                    f"Above diagonal = {config2_label} better\nBelow diagonal = {config1_label} better",
                    transform=ax.transAxes,
                    fontsize=16,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

                plt.tight_layout()

                if output_dir:
                    filepath = output_dir / "sgd_parameter_scatter_accuracy_gpt.pdf"
                    plt.savefig(filepath, bbox_inches="tight")
                    print(f"  Saved: {filepath}")
                else:
                    plt.show()

                plt.close()


def create_sgd_vs_apts_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create comparison plots for SGD (no overlap/batch increase) vs APTS_D (with overlap/batch increase)."""
    print("\n" + "=" * 80)
    print("GENERATING SGD VS APTS_D COMPARISON PLOTS")
    print("=" * 80)

    # Filter SGD with no overlap and no batch increase (overlap~0, batch_inc~1.0)
    sgd_baseline = df[
        (df["optimizer"] == "sgd")
        & (df["overlap"].notna())
        & (df["batch_inc_factor"].notna())
        & (abs(df["overlap"]) < 0.01)
        & (abs(df["batch_inc_factor"] - 1.0) < 0.01)
    ].copy()

    # Filter APTS_D with overlap and batch increase (overlap~0.33, batch_inc~1.5)
    apts_d_optimized = df[
        (df["optimizer"] == "apts_d")
        & (df["overlap"].notna())
        & (df["batch_inc_factor"].notna())
        & (abs(df["overlap"] - 0.33) < 0.01)
        & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
    ].copy()

    if len(sgd_baseline) == 0:
        print("\nNo SGD data with overlap=0, batch_inc_factor=1.0 found. Skipping...")
        return

    if len(apts_d_optimized) == 0:
        print(
            "\nNo APTS_D data with overlap=0.33, batch_inc_factor=1.5 found. Skipping..."
        )
        return

    print(f"\nFound {len(sgd_baseline)} SGD runs (overlap=0%, batch inc. factor=1.0)")
    print(
        f"Found {len(apts_d_optimized)} APTS_D runs (overlap=33%, batch inc. factor=1.5)"
    )

    # Check subdomain counts for APTS_D
    if "num_subdomains" in apts_d_optimized.columns:
        subdomain_counts = sorted(apts_d_optimized["num_subdomains"].dropna().unique())
        if len(subdomain_counts) > 0:
            print(
                f"  APTS_D subdomain counts included: {[int(x) for x in subdomain_counts]}"
            )

    # Combine the two datasets
    combined_df = pd.concat([sgd_baseline, apts_d_optimized])

    if combined_df["total_params"].notna().any():
        # Aggregate across trials
        agg_df = (
            combined_df.groupby(["optimizer", "total_params"])
            .agg(
                {
                    "final_loss": ["mean", "std"],
                    "final_accuracy": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in agg_df.columns.values
        ]

        sns.set_style("whitegrid")

        # Plot 1: Loss vs Network Size
        fig, ax = plt.subplots(figsize=(10, 6))

        for optimizer in ["sgd", "apts_d"]:
            opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values(
                "total_params"
            )
            if len(opt_df) == 0:
                continue

            label = (
                r"SGD (overlap=0\%, batch inc. factor=1.0)"
                if optimizer == "sgd"
                else r"$\mathrm{SAPTS}_D$ (overlap=33\%, batch inc. factor=1.5)"
            )

            ax.errorbar(
                opt_df["total_params"],
                opt_df["final_loss_mean"],
                yerr=opt_df["final_loss_std"],
                marker="o",
                capsize=5,
                label=label,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel(r"Number of parameters", fontsize=18)
        ax.set_ylabel(r"Final avg. loss", fontsize=18)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / "sgd_vs_apts_d_loss_gpt.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

        # Plot 2: Accuracy vs Network Size
        if combined_df["final_accuracy"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 6))

            for optimizer in ["sgd", "apts_d"]:
                opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values(
                    "total_params"
                )
                if len(opt_df) == 0:
                    continue

                label = (
                    r"SGD (overlap=0\%, batch inc. factor=1.0)"
                    if optimizer == "sgd"
                    else r"$\mathrm{SAPTS}_D$ (overlap=33\%, batch inc. factor=1.5)"
                )

                ax.errorbar(
                    opt_df["total_params"],
                    opt_df["final_accuracy_mean"],
                    yerr=opt_df["final_accuracy_std"],
                    marker="o",
                    capsize=5,
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

            ax.set_xlabel(r"Number of parameters", fontsize=18)
            ax.set_ylabel(r"Final avg. accuracy (\%)", fontsize=18)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            plt.tight_layout()

            if output_dir:
                filepath = output_dir / "sgd_vs_apts_d_accuracy_gpt.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()


def export_results(df: pd.DataFrame, output_path: Path) -> None:
    """Export results to CSV for further analysis."""
    # Drop the history column (too large for CSV)
    export_df = df.drop(columns=["history"])

    export_df.to_csv(output_path, index=False)
    print(f"\nExported results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameterization test results from wandb for GPT models"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ohtests",
        help="Wandb project name (default: ohtests)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="cruzas-universit-della-svizzera-italiana",
        help="Wandb entity (username or team). If not specified, uses your default entity.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of displaying them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hyperparam_analysis_gpt"),
        help="Directory to save plots and results (default: ./hyperparam_analysis_gpt)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--filter-optimizer",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific optimizers (e.g., --filter-optimizer sgd apts_d)",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default="nanogpt",
        help="Filter to specific model type (default: nanogpt)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of wandb data (forces fresh fetch)",
    )
    parser.add_argument(
        "--cache-max-age",
        type=int,
        default=24,
        help="Maximum age of cached data in hours (default: 24)",
    )

    parser.add_argument(
        "--sgd-overlap",
        type=float,
        default=0.0,
        help="Overlap value to use for SGD data in plots (default: 0.0 for no overlap)",
    )

    args = parser.parse_args()

    # Create output directory if saving plots
    if args.save_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    # Fetch runs from wandb
    filters = {}
    if args.filter_optimizer:
        filters["config.optimizer"] = {"$in": args.filter_optimizer}
    if args.filter_model:
        filters["config.model_name"] = args.filter_model

    runs = fetch_runs(
        project=args.project,
        entity=args.entity,
        filters=filters if filters else None,
        use_cache=not args.no_cache,
        cache_max_age_hours=args.cache_max_age,
    )

    if len(runs) == 0:
        print("\nNo runs found. Make sure:")
        print("  1. You've logged in to wandb (run: wandb login)")
        print("  2. The project name is correct")
        print("  3. You have access to the project")
        print("  4. Experiments have been run and logged to wandb")
        return

    # Extract data
    df = extract_run_data(runs)

    if len(df) == 0:
        print("\nNo valid runs found after filtering.")
        return

    # Validate APTS parameters
    validate_apts_parameters(df)

    # Print analyses
    print_summary_statistics(df)
    print_comparison_by_architecture(df)
    print_hyperparameterization_analysis(df)

    # Create plots
    if args.save_plots:
        output_dir = args.output_dir
    else:
        output_dir = None

    try:
        # create_convergence_plots(df, output_dir)
        create_comparison_plots(df, output_dir, sgd_overlap=args.sgd_overlap)
        create_sgd_parameter_comparison_plots_gpt(df, output_dir)
        # create_sgd_vs_apts_comparison_plots(df, output_dir)  # Removed - redundant with comparison plots
    except Exception as e:
        print(f"\nWarning: Error creating plots: {e}")
        print("Continuing with text analysis...")

    # Export to CSV if requested
    if args.export_csv:
        csv_path = args.output_dir / "hyperparam_results_gpt.csv"
        export_results(df, csv_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print(
        "  - Check how final loss/accuracy changes with model size for each optimizer"
    )
    print("  - Look for optimizers that maintain performance as models grow")
    print("  - Compare convergence speed (iterations to reach target performance)")
    print("  - Examine runtime efficiency vs. final performance trade-offs")

    if args.save_plots:
        print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
