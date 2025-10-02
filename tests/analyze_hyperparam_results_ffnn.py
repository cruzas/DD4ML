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

# Enable LaTeX rendering
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

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
                cache_data = {
                    "_fetched_at": time.time(),
                    "runs": runs_list
                }
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


def format_optimizer_name(optimizer: str) -> str:
    """Format optimizer name for display with LaTeX."""
    formatter = {
        "sgd": r"SGD",
        "apts_d": r"$\mathrm{SAPTS}_D$",
        "apts": r"$\mathrm{SAPTS}$",
    }
    return formatter.get(optimizer.lower(), optimizer)


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
        ax.set_ylabel(r"Avg. loss", fontsize=12)
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
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create comparison plots across network sizes."""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    sns.set_style("whitegrid")

    # Aggregate across trials and architectures (group by total_params only)
    agg_df = (
        df.groupby(["optimizer", "total_params"])
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
        "_".join(col).strip("_") if col[1] else col[0] for col in agg_df.columns.values
    ]

    # Plot 1: Final Loss vs Network Size
    fig, ax = plt.subplots(figsize=(10, 6))

    for optimizer in sorted(agg_df["optimizer"].unique()):
        opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values("total_params")
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

    plt.tight_layout()

    if output_dir:
        filepath = output_dir / "loss_vs_network_size_ffnn.pdf"
        plt.savefig(filepath, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    else:
        plt.show()

    plt.close()

    # Plot 2: Final Accuracy vs Network Size
    fig, ax = plt.subplots(figsize=(10, 6))

    for optimizer in sorted(agg_df["optimizer"].unique()):
        opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values("total_params")
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

    plt.tight_layout()

    if output_dir:
        filepath = output_dir / "accuracy_vs_network_size_ffnn.pdf"
        plt.savefig(filepath, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    else:
        plt.show()

    plt.close()

    # Plot 3: Heatmap of loss by width and depth
    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]

        # Create pivot table for loss
        pivot = opt_df.pivot_table(
            values="final_loss", index="num_layers", columns="width", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": r"Avg. loss"},
        )
        ax.set_title(
            f"{format_optimizer_name(optimizer)}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel(r"Width", fontsize=12)
        ax.set_ylabel(r"Depth", fontsize=12)

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"heatmap_loss_{optimizer}_ffnn.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    # Plot 4: Heatmap of accuracy by width and depth
    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]

        # Create pivot table for accuracy
        pivot = opt_df.pivot_table(
            values="final_accuracy",
            index="num_layers",
            columns="width",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": r"Final avg. accuracy (\%)"},
        )
        ax.set_title(
            f"{format_optimizer_name(optimizer)}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel(r"Width", fontsize=12)
        ax.set_ylabel(r"Depth", fontsize=12)

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"heatmap_accuracy_{optimizer}_ffnn.pdf"
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
        create_convergence_plots(df, output_dir)
        create_comparison_plots(df, output_dir)
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
