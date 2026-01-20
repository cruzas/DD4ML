#!/usr/bin/env python3
"""
Analysis Script for Hyperparameterization Tests

This script analyzes results from hyperparameterization_test.py experiments
stored in wandb under the 'ohtests' project. It compares SGD vs APTS_D (and
other optimizers) across different network sizes (width and depth).

Usage:
    python analyze_hyperparam_results.py
    python analyze_hyperparam_results.py --entity your-wandb-username
    python analyze_hyperparam_results.py --save-plots
    python analyze_hyperparam_results.py --metric loss
"""

#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

# =============================================================================
# MODERN COLOR SCHEME & BEAMER STYLING
# =============================================================================
COLORS = {
    "modernBlue": "#0054A6",
    "modernLight": "#2980B9",
    "modernBlack": "#1E1E1E",
    "modernDark": "#34495E",
    "modernPink": "#FF2D55",
    "modernPurple": "#8E44AD",
    "modernTeal": "#00A896",
    "background": "#FAFAFA",
}

custom_cycler = cycler(
    color=[
        COLORS["modernBlue"],
        COLORS["modernPink"],
        COLORS["modernTeal"],
        COLORS["modernPurple"],
        COLORS["modernLight"],
        COLORS["modernDark"],
    ]
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{sfmath}",
        "font.family": "sans-serif",
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 30,
        "axes.prop_cycle": custom_cycler,
        "text.color": COLORS["modernBlack"],
        "axes.labelcolor": COLORS["modernDark"],
        "xtick.color": COLORS["modernDark"],
        "ytick.color": COLORS["modernDark"],
        "axes.edgecolor": COLORS["modernDark"],
        "grid.color": "#BDC3C7",
        "lines.linewidth": 4,
        "lines.markersize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.facecolor": COLORS["background"],
    }
)


def get_custom_heat_cmap(metric_type="loss"):
    if metric_type == "loss":
        colors = [COLORS["modernBlue"], "#FFFFFF", COLORS["modernPink"]]
    else:
        colors = [COLORS["modernPink"], "#FFFFFF", COLORS["modernBlue"]]
    return mcolors.LinearSegmentedColormap.from_list("modern_theme", colors)


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

        # Extract network architecture info
        width = config.get("width", None)
        num_layers = config.get("num_layers", None)

        # Convert to int if they are not None and not already int
        try:
            width = int(width) if width is not None else None
        except (ValueError, TypeError):
            width = None

        try:
            num_layers = int(num_layers) if num_layers is not None else None
        except (ValueError, TypeError):
            num_layers = None

        # Calculate total parameters (approximate for FFNN)
        if width and num_layers:
            # Input layer: 784 (MNIST) -> width
            # Hidden layers: width -> width (num_layers - 1 times)
            # Output layer: width -> 10
            total_params = (
                (784 * width) + ((num_layers - 1) * width * width) + (width * 10)
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
                "width": width,
                "num_layers": num_layers,
                "total_params": total_params,
                "trial": trial,
                "epochs": epochs,
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
    df = df.dropna(subset=["optimizer", "width", "num_layers"])
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
        print(
            f"  Avg final accuracy: {opt_df['final_accuracy'].mean():.4f} ± {opt_df['final_accuracy'].std():.4f}"
        )
        if opt_df["total_runtime"].notna().any():
            print(
                f"  Avg runtime: {opt_df['total_runtime'].mean():.2f}s ± {opt_df['total_runtime'].std():.2f}s"
            )
        if opt_df["total_grad_evals"].notna().any():
            print(f"  Avg grad evals: {opt_df['total_grad_evals'].mean():.0f}")


def print_comparison_by_network_size(df: pd.DataFrame) -> None:
    """Compare optimizers across different network sizes."""
    print("\n" + "=" * 80)
    print("COMPARISON BY NETWORK SIZE")
    print("=" * 80)

    # Group by network architecture
    architectures = (
        df.groupby(["width", "num_layers"])
        .size()
        .reset_index()[["width", "num_layers"]]
    )

    for _, arch in architectures.iterrows():
        width = arch["width"]
        num_layers = arch["num_layers"]

        arch_df = df[(df["width"] == width) & (df["num_layers"] == num_layers)]

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
                    "final_accuracy": ["mean", "std", "max"],
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
            opt_df.groupby(["width", "num_layers", "total_params"])
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
            f"{'Params':<12} {'Width':<8} {'Depth':<8} {'Loss (mean±std)':<20} {'Accuracy (mean±std)':<20}"
        )
        print("─" * 70)

        for _, row in summary.iterrows():
            params = f"{int(row['total_params']):,}"
            width = int(row["width"])
            depth = int(row["num_layers"])
            loss_mean = row[("final_loss", "mean")]
            loss_std = row[("final_loss", "std")]
            acc_mean = row[("final_accuracy", "mean")]
            acc_std = row[("final_accuracy", "std")]

            print(
                f"{params:<12} {width:<8} {depth:<8} {loss_mean:.6f}±{loss_std:.6f}    {acc_mean:.4f}±{acc_std:.4f}"
            )


def create_convergence_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create convergence plots for different optimizers and network sizes."""
    print("\n" + "=" * 80)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)

    # Get unique architectures
    architectures = (
        df.groupby(["width", "num_layers"])
        .size()
        .reset_index()[["width", "num_layers"]]
    )

    for _, arch in architectures.iterrows():
        width = arch["width"]
        num_layers = arch["num_layers"]

        arch_df = df[(df["width"] == width) & (df["num_layers"] == num_layers)]

        if len(arch_df) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Width={width}, Depth={num_layers}", fontsize=16, fontweight="bold"
        )

        # Plot loss convergence
        ax = axes[0]
        for optimizer in sorted(arch_df["optimizer"].unique()):
            opt_runs = arch_df[arch_df["optimizer"] == optimizer]

            # Collect all history data for this optimizer across trials
            all_histories = []
            for _, run_data in opt_runs.iterrows():
                history = run_data["history"]
                if (
                    history is not None
                    and not history.empty
                    and "epoch" in history.columns
                    and "loss" in history.columns
                ):
                    all_histories.append(history[["epoch", "loss"]])

            # Average across trials
            if all_histories:
                combined_history = pd.concat(all_histories)
                epoch_data = combined_history.groupby("epoch")["loss"].agg(
                    ["mean", "std"]
                )
                ax.plot(
                    epoch_data.index,
                    epoch_data["mean"],
                    label=format_optimizer_name(optimizer),
                    linewidth=2,
                )
                # Add confidence interval
                ax.fill_between(
                    epoch_data.index,
                    epoch_data["mean"] - epoch_data["std"],
                    epoch_data["mean"] + epoch_data["std"],
                    alpha=0.2,
                )

        ax.set_xlabel(r"Epoch", fontsize=12)
        ax.set_ylabel(r"Final avg. loss", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot accuracy convergence
        ax = axes[1]
        for optimizer in sorted(arch_df["optimizer"].unique()):
            opt_runs = arch_df[arch_df["optimizer"] == optimizer]

            # Collect all history data for this optimizer across trials
            all_histories = []
            for _, run_data in opt_runs.iterrows():
                history = run_data["history"]
                if (
                    history is not None
                    and not history.empty
                    and "epoch" in history.columns
                    and "accuracy" in history.columns
                ):
                    all_histories.append(history[["epoch", "accuracy"]])

            # Average across trials
            if all_histories:
                combined_history = pd.concat(all_histories)
                epoch_data = combined_history.groupby("epoch")["accuracy"].agg(
                    ["mean", "std"]
                )
                ax.plot(
                    epoch_data.index,
                    epoch_data["mean"],
                    label=format_optimizer_name(optimizer),
                    linewidth=2,
                )
                # Add confidence interval
                ax.fill_between(
                    epoch_data.index,
                    epoch_data["mean"] - epoch_data["std"],
                    epoch_data["mean"] + epoch_data["std"],
                    alpha=0.2,
                )

        ax.set_xlabel(r"Epoch", fontsize=12)
        ax.set_ylabel(r"Avg. accuracy (\%)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            filename = f"convergence_w{width}_nl{num_layers}_ffnn.pdf"
            filepath = output_dir / filename
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()


def create_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None, sgd_overlap: float = 0.0
) -> None:
    """Create comparison plots across network sizes, separated by parameter combinations.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots (if None, shows plots instead)
        sgd_overlap: Overlap value to use for SGD data (default: 0.0 for no overlap)
    """
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    print(f"Using SGD with overlap={sgd_overlap:.2f}")

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
        # For APTS_D and APTS_P, also group by num_subdomains
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

        # Plot 1: Final Loss vs Network Size
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
                bbox=dict(
                    boxstyle="round",
                    facecolor=COLORS["background"],
                    edgecolor=COLORS["modernDark"],
                    alpha=0.8,
                ),
            )

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"loss_vs_network_size_ffnn{suffix}.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

        # Plot 2: Final Accuracy vs Network Size
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
                bbox=dict(
                    boxstyle="round",
                    facecolor=COLORS["background"],
                    edgecolor=COLORS["modernDark"],
                    alpha=0.8,
                ),
            )

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"accuracy_vs_network_size_ffnn{suffix}.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    # Plot 3: Heatmap of loss by width and depth
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
        pivot = sgd_selected.pivot_table(
            values="final_loss", index="num_layers", columns="width", aggfunc="mean"
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

    all_acc_pivots_global = []

    # For SGD, use specified overlap value
    if len(sgd_selected) > 0:
        pivot = sgd_selected.pivot_table(
            values="final_accuracy", index="num_layers", columns="width", aggfunc="mean"
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
            pivot = sub_df.pivot_table(
                values="final_accuracy",
                index="num_layers",
                columns="width",
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
            pivot = sgd_selected.pivot_table(
                values="final_loss",
                index="num_layers",
                columns="width",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append(
                    {
                        "pivot": pivot,
                        "title": format_optimizer_name("sgd"),
                        "optimizer": "sgd",
                        "num_subs": None,
                    }
                )

        # Add APTS variants with overlap=0.33, batch_inc=1.5
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
                pivot = sub_df.pivot_table(
                    values="final_loss",
                    index="num_layers",
                    columns="width",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append(
                        {
                            "pivot": pivot,
                            "title": format_optimizer_name(optimizer, int(num_subs)),
                            "optimizer": optimizer,
                            "num_subs": int(num_subs),
                        }
                    )

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
                    cmap=get_custom_heat_cmap("loss"),
                    ax=ax,
                    vmin=global_loss_min,
                    vmax=global_loss_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(
                        col == 0
                    ),  # Show y tick labels only on leftmost column
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
                axes[idx].axis("off")

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=global_loss_min, vmax=global_loss_max)
            sm = cm.ScalarMappable(cmap=get_custom_heat_cmap("loss"), norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. loss", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_loss_combined_ffnn.pdf"
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
            pivot = sgd_selected.pivot_table(
                values="final_accuracy",
                index="num_layers",
                columns="width",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append(
                    {
                        "pivot": pivot,
                        "title": format_optimizer_name("sgd"),
                        "optimizer": "sgd",
                        "num_subs": None,
                    }
                )

        # Add APTS variants with overlap=0.33, batch_inc=1.5
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
                pivot = sub_df.pivot_table(
                    values="final_accuracy",
                    index="num_layers",
                    columns="width",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append(
                        {
                            "pivot": pivot,
                            "title": format_optimizer_name(optimizer, int(num_subs)),
                            "optimizer": optimizer,
                            "num_subs": int(num_subs),
                        }
                    )

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
                    cmap=get_custom_heat_cmap("accuracy"),
                    ax=ax,
                    vmin=global_acc_min,
                    vmax=global_acc_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(
                        col == 0
                    ),  # Show y tick labels only on leftmost column
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
                axes[idx].axis("off")

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=global_acc_min, vmax=global_acc_max)
            sm = cm.ScalarMappable(cmap=get_custom_heat_cmap("accuracy"), norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. accuracy (\%)", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_accuracy_combined_ffnn.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()


def create_sgd_parameter_comparison_plots(
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

    # Sort combinations to ensure consistent axis assignment:
    # We want config with overlap=0.0 on x-axis (Config 1)
    # and config with overlap=0.33 on y-axis (Config 2)
    param_combinations.sort(key=lambda x: x[0])

    overlap1, batch_inc1 = param_combinations[0]
    overlap2, batch_inc2 = param_combinations[1]

    label1 = f"overlap={overlap1*100:.0f}\\%, batch inc. factor={batch_inc1:.2f}"
    label2 = f"overlap={overlap2*100:.0f}\\%, batch inc. factor={batch_inc2:.2f}"

    sns.set_style("whitegrid")

    # --- Plot 1 & 2: Line plots (unchanged logic, but using sorted labels) ---
    for metric in ["loss", "accuracy"]:
        if metric == "accuracy" and not sgd_df["final_accuracy"].notna().any():
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for ov, bif in param_combinations:
            subset = sgd_df[
                (abs(sgd_df["overlap"] - ov) < 0.01)
                & (abs(sgd_df["batch_inc_factor"] - bif) < 0.01)
            ]
            if len(subset) == 0:
                continue
            agg = (
                subset.groupby("total_params")
                .agg({f"final_{metric}": ["mean", "std"]})
                .reset_index()
            )
            agg.columns = ["total_params", "m", "s"]
            ax.errorbar(
                agg["total_params"],
                agg["m"],
                yerr=agg["s"],
                marker="o",
                capsize=5,
                label=f"overlap={ov*100:.0f}\\%, bif={bif:.2f}",
                linewidth=2,
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"Number of parameters")
        ax.set_ylabel(f"Final avg. {metric} (SGD)")
        ax.legend()
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f"sgd_parameter_comparison_{metric}_ffnn.pdf")
        plt.close()

    # --- Plot 3 & 4: Scatter plots (Update for Axis Consistency) ---
    sgd1 = sgd_df[
        (abs(sgd_df["overlap"] - overlap1) < 0.01)
        & (abs(sgd_df["batch_inc_factor"] - batch_inc1) < 0.01)
    ].copy()
    sgd2 = sgd_df[
        (abs(sgd_df["overlap"] - overlap2) < 0.01)
        & (abs(sgd_df["batch_inc_factor"] - batch_inc2) < 0.01)
    ].copy()

    sgd1_agg = (
        sgd1.groupby(["width", "num_layers"])
        .agg({"final_loss": "mean", "final_accuracy": "mean", "total_params": "first"})
        .reset_index()
    )
    sgd2_agg = (
        sgd2.groupby(["width", "num_layers"])
        .agg({"final_loss": "mean", "final_accuracy": "mean", "total_params": "first"})
        .reset_index()
    )

    merged = pd.merge(
        sgd1_agg,
        sgd2_agg,
        on=["width", "num_layers", "total_params"],
        suffixes=("_1", "_2"),
    )

    if len(merged) > 0:
        for metric in ["loss", "accuracy"]:
            m1, m2 = f"final_{metric}_1", f"final_{metric}_2"
            if merged[m1].isna().all():
                continue

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged[m1],
                merged[m2],
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
            )

            # Diagonal line
            lims = [
                min(merged[m1].min(), merged[m2].min()),
                max(merged[m1].max(), merged[m2].max()),
            ]
            ax.plot(lims, lims, "k--", alpha=0.5)

            ax.set_xlabel(f"Final {metric}: {label1}", fontsize=18)
            ax.set_ylabel(f"Final {metric}: {label2}", fontsize=18)

            # Helper for label text
            def get_cfg_desc(ov, bif):
                return (
                    "overlap \\& batch inc."
                    if ov > 0
                    else "no overlap \\& no batch inc."
                )

            better_side = "Below" if metric == "loss" else "Above"
            other_side = "Above" if metric == "loss" else "Below"

            ax.text(
                0.05,
                0.95,
                f"{better_side} diagonal = {get_cfg_desc(overlap2, batch_inc2)} better\n"
                f"{other_side} diagonal = {get_cfg_desc(overlap1, batch_inc1)} better",
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=COLORS["background"], alpha=0.5),
            )

            plt.tight_layout()
            if output_dir:
                plt.savefig(output_dir / f"sgd_parameter_scatter_{metric}_ffnn.pdf")
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
            filepath = output_dir / "sgd_vs_apts_d_loss_ffnn.pdf"
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
                filepath = output_dir / "sgd_vs_apts_d_accuracy_ffnn.pdf"
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
        description="Analyze hyperparameterization test results from wandb"
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
        default=Path("./hyperparam_analysis"),
        help="Directory to save plots and results (default: ./hyperparam_analysis)",
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
        default="medium_ffnn",
        help="Filter to specific model type (default: medium_ffnn)",
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
    print_comparison_by_network_size(df)
    print_hyperparameterization_analysis(df)

    # Create plots
    if args.save_plots:
        output_dir = args.output_dir
    else:
        output_dir = None

    try:
        # create_convergence_plots(df, output_dir)
        create_comparison_plots(df, output_dir, sgd_overlap=args.sgd_overlap)
        create_sgd_parameter_comparison_plots(df, output_dir)
        # create_sgd_vs_apts_comparison_plots(df, output_dir)  # Removed - redundant with comparison plots
    except Exception as e:
        print(f"\nWarning: Error creating plots: {e}")
        print("Continuing with text analysis...")

    # Export to CSV if requested
    if args.export_csv:
        csv_path = args.output_dir / "hyperparam_results.csv"
        export_results(df, csv_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print(
        "  - Check how final loss/accuracy changes with network size for each optimizer"
    )
    print("  - Look for optimizers that maintain performance as networks grow")
    print("  - Compare convergence speed (epochs to reach target performance)")
    print("  - Examine runtime efficiency vs. final performance trade-offs")

    if args.save_plots:
        print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
