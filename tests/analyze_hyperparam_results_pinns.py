#!/usr/bin/env python3
"""
Analysis Script for PINN Hyperparameterization Tests

This script analyzes results from hyperparameterization_test_pinns.py experiments
stored in wandb under the 'ohtests' project. It compares SGD vs APTS optimizers
across different PINN network sizes and datasets.

Usage:
    python analyze_hyperparam_results_pinns.py
    python analyze_hyperparam_results_pinns.py --entity your-wandb-username
    python analyze_hyperparam_results_pinns.py --save-plots
    python analyze_hyperparam_results_pinns.py --filter-dataset poisson2d
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
    cache_dir = Path.home() / ".cache" / "dd4ml_hyperparam_analysis"
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
        "apts_pinn": r"$\mathrm{SAPTS}_{PINN}$",
    }
    base_name = formatter.get(optimizer.lower(), optimizer)

    # Add subdomain count for APTS variants if specified
    if num_subdomains is not None and optimizer.lower() in [
        "apts_d",
        "apts_p",
        "apts_ip",
        "apts_pinn",
    ]:
        base_name += f" ($N={num_subdomains}$)"

    return base_name


def format_dataset_name(dataset: str) -> str:
    """Format dataset name for display with LaTeX."""
    formatter = {
        "poisson1d": r"Poisson 1D",
        "poisson2d": r"Poisson 2D",
        "poisson3d": r"Poisson 3D",
        "allencahn1d": r"Allen-Cahn 1D",
        "allencahn1d_time": r"Allen-Cahn 1D (Time)",
    }
    return formatter.get(dataset.lower(), dataset)


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


def validate_apts_parameters(df: pd.DataFrame) -> None:
    """Validate that APTS variants have overlap=0.33 and batch_inc_factor=1.5."""
    print("\n" + "=" * 80)
    print("VALIDATING APTS PARAMETERS")
    print("=" * 80)

    apts_variants = ["apts_d", "apts_p", "apts_ip", "apts_pinn"]
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


def calculate_pinn_parameters(
    width: int, num_layers: int, input_dim: int = 2, output_dim: int = 1
) -> int:
    """Calculate total parameters for a PINN network.

    Args:
        width: Width of hidden layers
        num_layers: Number of layers (including hidden layers)
        input_dim: Input dimension (1 for 1D, 2 for 2D, 3 for 3D problems)
        output_dim: Output dimension (typically 1 for scalar PDEs)
    """
    # Input layer: input_dim -> width
    # Hidden layers: width -> width (num_layers - 1 times)
    # Output layer: width -> output_dim
    total_params = (
        (input_dim * width) + ((num_layers - 1) * width * width) + (width * output_dim)
    )
    return total_params


def infer_input_dim(dataset_name: str) -> int:
    """Infer input dimension from dataset name."""
    if "1d" in dataset_name.lower():
        return 1
    elif "2d" in dataset_name.lower():
        return 2
    elif "3d" in dataset_name.lower():
        return 3
    else:
        return 2  # Default to 2D


def extract_run_data(runs: List) -> pd.DataFrame:
    """Extract relevant data from wandb runs into a DataFrame."""
    results = []

    print(f"\nProcessing {len(runs)} runs...")

    for run in runs:
        config = run.config
        summary = run.summary._json_dict

        # Extract network architecture info
        width = config.get("width", None)
        num_layers = config.get("num_layers", None)
        dataset_name = config.get("dataset_name", "poisson2d")

        # Convert to int if they are not None and not already int
        try:
            width = int(width) if width is not None else None
        except (ValueError, TypeError):
            width = None

        try:
            num_layers = int(num_layers) if num_layers is not None else None
        except (ValueError, TypeError):
            num_layers = None

        # Calculate total parameters for PINN
        if width and num_layers:
            input_dim = infer_input_dim(dataset_name)
            total_params = calculate_pinn_parameters(
                width, num_layers, input_dim, output_dim=1
            )
        else:
            total_params = None

        # Extract optimizer info
        optimizer = config.get("optimizer", "unknown")

        # Extract trial number
        trial = config.get("trial", 1)

        # Extract training hyperparameters
        epochs = config.get("epochs", None)
        batch_size = config.get("batch_size", None)
        learning_rate = config.get("learning_rate", None)
        num_subdomains = config.get("num_subdomains", None)
        overlap = config.get("overlap", None)
        batch_inc_factor = config.get("batch_inc_factor", None)

        # Extract final metrics
        final_loss = summary.get("loss", None)
        total_runtime = summary.get("running_time", None)
        total_grad_evals = summary.get("grad_evals", None)

        # Get history for convergence analysis (with caching)
        history = _load_history_cached(run.id, run)

        results.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "optimizer": optimizer,
                "dataset_name": dataset_name,
                "width": width,
                "num_layers": num_layers,
                "input_dim": input_dim if width and num_layers else None,
                "total_params": total_params,
                "trial": trial,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_subdomains": num_subdomains,
                "overlap": overlap,
                "batch_inc_factor": batch_inc_factor,
                "final_loss": final_loss,
                "total_runtime": total_runtime,
                "total_grad_evals": total_grad_evals,
                "history": history,
                "run_url": run.url,
            }
        )

    df = pd.DataFrame(results)

    # Filter out runs missing critical info
    initial_count = len(df)
    df = df.dropna(subset=["optimizer", "width", "num_layers", "dataset_name"])
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
    print(f"Datasets: {sorted(df['dataset_name'].unique())}")
    print(f"Network widths: {sorted(df['width'].unique())}")
    print(f"Network depths: {sorted(df['num_layers'].unique())}")
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
        if opt_df["total_runtime"].notna().any():
            print(
                f"  Avg runtime: {opt_df['total_runtime'].mean():.2f}s ± {opt_df['total_runtime'].std():.2f}s"
            )
        if opt_df["total_grad_evals"].notna().any():
            print(f"  Avg grad evals: {opt_df['total_grad_evals'].mean():.0f}")


def print_comparison_by_network_size(df: pd.DataFrame) -> None:
    """Compare optimizers across different network sizes and datasets."""
    print("\n" + "=" * 80)
    print("COMPARISON BY NETWORK SIZE AND DATASET")
    print("=" * 80)

    # Group by dataset and network architecture
    for dataset in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset]

        print(f"\n{'═' * 80}")
        print(f"Dataset: {format_dataset_name(dataset)}")
        print(f"{'═' * 80}")

        architectures = (
            dataset_df.groupby(["width", "num_layers"])
            .size()
            .reset_index()[["width", "num_layers"]]
        )

        for _, arch in architectures.iterrows():
            width = arch["width"]
            num_layers = arch["num_layers"]

            arch_df = dataset_df[
                (dataset_df["width"] == width)
                & (dataset_df["num_layers"] == num_layers)
            ]

            if len(arch_df) == 0:
                continue

            print(f"\n{'─' * 80}")
            print(
                f"Width={width}, Depth={num_layers} ({arch_df['total_params'].iloc[0]:,} parameters)"
            )
            print(f"{'─' * 80}")

            # Compare optimizers for this architecture
            comparison = (
                arch_df.groupby("optimizer")
                .agg(
                    {
                        "final_loss": ["mean", "std", "min"],
                        "total_runtime": ["mean", "std"],
                        "total_grad_evals": ["mean", "std"],
                    }
                )
                .round(6)
            )

            print(comparison.to_string())


def print_hyperparameterization_analysis(df: pd.DataFrame) -> None:
    """Analyze how network size affects each optimizer."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETERIZATION EFFECT ANALYSIS")
    print("=" * 80)
    print("\nHow does increasing network size affect each optimizer?\n")

    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]

        print(f"\n{optimizer.upper()}:")
        print(f"{'─' * 60}")

        # Sort by total parameters
        summary = (
            opt_df.groupby(["dataset_name", "width", "num_layers", "total_params"])
            .agg(
                {
                    "final_loss": ["mean", "std"],
                }
            )
            .reset_index()
            .sort_values("total_params")
        )

        # Flatten column names for easier access
        summary.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        print(
            f"{'Dataset':<20} {'Params':<12} {'Width':<8} {'Depth':<8} {'Loss (mean±std)':<20}"
        )
        print("─" * 70)

        for _, row in summary.iterrows():
            dataset = row["dataset_name"]
            params = f"{int(row['total_params']):,}"
            width = int(row["width"])
            depth = int(row["num_layers"])
            loss_mean = row["final_loss_mean"]
            loss_std = row["final_loss_std"]

            print(
                f"{dataset:<20} {params:<12} {width:<8} {depth:<8} {loss_mean:.6f}±{loss_std:.6f}"
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
    print("GENERATING COMPARISON PLOTS across network sizes, separated by dataset." "")
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    sns.set_style("whitegrid")

    # Group by dataset
    for dataset in sorted(df["dataset_name"].unique()):
        dataset_all = df[df["dataset_name"] == dataset]

        print(f"\n--- Creating plots for {format_dataset_name(dataset)} ---")

        # Filter to use SGD with best performing configuration (no overlap)
        # and APTS with overlap/batch_inc_factor
        sgd_subset = dataset_all[
            (dataset_all["optimizer"] == "sgd")
            & (abs(dataset_all["overlap"]) < 0.01)
            & (abs(dataset_all["batch_inc_factor"] - 1.0) < 0.01)
        ]

        apts_subset = dataset_all[
            dataset_all["optimizer"].isin(["apts_d", "apts_p", "apts_ip", "apts_pinn"])
        ]

        # Combine SGD (no overlap) with APTS variants
        dataset_df = pd.concat([sgd_subset, apts_subset])

        if len(dataset_df) == 0:
            print(f"  No data for {dataset}, skipping...")
            continue

        # Aggregate across trials
        groupby_cols = ["optimizer", "total_params"]

        # Add num_subdomains to grouping for APTS variants
        apts_variants = ["apts_d", "apts_p", "apts_ip", "apts_pinn"]
        if any(opt in dataset_df["optimizer"].unique() for opt in apts_variants):
            groupby_cols.append("num_subdomains")

        agg_df = (
            dataset_df.groupby(groupby_cols)
            .agg(
                {
                    "final_loss": ["mean", "std"],
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

        # Plot: Final Loss vs Network Size
        fig, ax = plt.subplots(figsize=(10, 6))

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
        ax.set_ylabel(r"Final loss", fontsize=12)
        ax.set_title(f"{format_dataset_name(dataset)}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Add parameter values as text annotation
        # Check if APTS data has overlap/batch_inc_factor values
        apts_data = dataset_df[
            dataset_df["optimizer"].isin(["apts_d", "apts_p", "apts_ip", "apts_pinn"])
        ]
        if len(apts_data) > 0:
            overlap_vals = apts_data["overlap"].dropna().unique()
            batch_inc_vals = apts_data["batch_inc_factor"].dropna().unique()
            if len(overlap_vals) > 0 and len(batch_inc_vals) > 0:
                param_text = (
                    f"APTS: ov.={overlap_vals[0]:.2f}, bif={batch_inc_vals[0]:.2f}\n"
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
            filepath = output_dir / f"loss_vs_network_size_{dataset}_pinn.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()


def create_heatmaps(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """Create combined heatmaps of loss by width and depth for each dataset."""
    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS")
    print("=" * 80)

    # Calculate global min/max across ALL data for consistent scaling
    print("\n" + "=" * 80)
    print("CALCULATING GLOBAL HEATMAP SCALES")
    print("=" * 80)

    all_loss_pivots_global = []
    apts_variants = ["apts_d", "apts_p", "apts_ip", "apts_pinn"]

    for dataset in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset]

        # SGD data
        sgd_df = dataset_df[dataset_df["optimizer"] == "sgd"]
        if len(sgd_df) > 0:
            pivot = sgd_df.pivot_table(
                values="final_loss",
                index="num_layers",
                columns="width",
                aggfunc="mean",
            )
            if not pivot.empty:
                all_loss_pivots_global.append(pivot)

        # APTS variants
        for optimizer in apts_variants:
            opt_df = dataset_df[dataset_df["optimizer"] == optimizer]
            for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                pivot = sub_df.pivot_table(
                    values="final_loss",
                    index="num_layers",
                    columns="width",
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

    # Generate combined heatmaps for each dataset
    for dataset in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset]

        print(f"\n--- Creating combined heatmap for {format_dataset_name(dataset)} ---")

        if global_loss_min is not None and global_loss_max is not None:
            heatmap_data = []

            # Add SGD
            sgd_df = dataset_df[dataset_df["optimizer"] == "sgd"]
            if len(sgd_df) > 0:
                pivot = sgd_df.pivot_table(
                    values="final_loss",
                    index="num_layers",
                    columns="width",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append({
                        "pivot": pivot,
                        "title": format_optimizer_name("sgd"),
                        "optimizer": "sgd",
                        "num_subs": None
                    })

            # Add APTS variants
            for optimizer in apts_variants:
                opt_df = dataset_df[dataset_df["optimizer"] == optimizer]
                for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                    sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                    pivot = sub_df.pivot_table(
                        values="final_loss",
                        index="num_layers",
                        columns="width",
                        aggfunc="mean",
                    )
                    if not pivot.empty:
                        heatmap_data.append({
                            "pivot": pivot,
                            "title": format_optimizer_name(optimizer, int(num_subs)),
                            "optimizer": optimizer,
                            "num_subs": int(num_subs)
                        })

            # Create combined figure with all heatmaps
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
                    ax.set_title(data["title"], fontsize=14, fontweight="bold")

                    # Only show x-label on center plot of bottom row
                    if row == n_rows - 1 and col == n_cols // 2:
                        ax.set_xlabel(r"Width", fontsize=12)
                    else:
                        ax.set_xlabel("")

                    # Only show y-label on leftmost column
                    if col == 0:
                        ax.set_ylabel(r"Depth", fontsize=12)
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
                cbar.set_label(r"Final loss", fontsize=14)

                if output_dir:
                    filepath = output_dir / f"heatmap_loss_combined_{dataset}_pinn.pdf"
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
        description="Analyze PINN hyperparameterization test results from wandb"
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
        default=Path("./hyperparam_analysis_pinns"),
        help="Directory to save plots and results (default: ./hyperparam_analysis_pinns)",
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
        "--filter-dataset",
        type=str,
        default=None,
        help="Filter to specific dataset (e.g., poisson2d)",
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
    filters = {"config.model_name": "pinn_ffnn"}
    if args.filter_optimizer:
        filters["config.optimizer"] = {"$in": args.filter_optimizer}
    if args.filter_dataset:
        filters["config.dataset_name"] = args.filter_dataset

    runs = fetch_runs(
        project=args.project,
        entity=args.entity,
        filters=filters,
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
    print_comparison_by_network_size(df)
    print_hyperparameterization_analysis(df)

    # Create plots
    if args.save_plots:
        output_dir = args.output_dir
    else:
        output_dir = None

    try:
        create_comparison_plots(df, output_dir, sgd_overlap=args.sgd_overlap)
        create_heatmaps(df, output_dir)
    except Exception as e:
        print(f"\nWarning: Error creating plots: {e}")
        print("Continuing with text analysis...")

    # Export to CSV if requested
    if args.export_csv:
        csv_path = args.output_dir / "hyperparam_results_pinns.csv"
        export_results(df, csv_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  - Check how final loss changes with network size for each optimizer")
    print("  - Look for optimizers that maintain performance as networks grow")
    print("  - Compare convergence speed (epochs to reach target performance)")
    print("  - Examine runtime efficiency vs. final performance trade-offs")
    print("  - Compare performance across different PINN datasets")

    if args.save_plots:
        print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
