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
        "apts_p": r"$\mathrm{SAPTS}_P$",
        "apts_ip": r"$\mathrm{SAPTS}_{IP}$",
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


def create_convergence_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create convergence plots for different optimizers and architectures."""
    print("\n" + "=" * 80)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)

    # Get unique architectures
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

        # Determine if we have accuracy data
        has_accuracy = arch_df["history"].apply(
            lambda h: h is not None and not h.empty and "accuracy" in h.columns
        ).any()

        if has_accuracy:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Embd={n_embd}, Head={n_head}, Layer={n_layer}", fontsize=16, fontweight="bold"
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            axes = [ax]
            fig.suptitle(
                f"Embd={n_embd}, Head={n_head}, Layer={n_layer}", fontsize=16, fontweight="bold"
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
                    and "_step" in history.columns
                    and "loss" in history.columns
                ):
                    all_histories.append(history[["_step", "loss"]])

            # Average across trials
            if all_histories:
                combined_history = pd.concat(all_histories)
                step_data = combined_history.groupby("_step")["loss"].agg(
                    ["mean", "std"]
                )
                ax.plot(
                    step_data.index,
                    step_data["mean"],
                    label=format_optimizer_name(optimizer),
                    linewidth=2,
                )
                # Add confidence interval
                ax.fill_between(
                    step_data.index,
                    step_data["mean"] - step_data["std"],
                    step_data["mean"] + step_data["std"],
                    alpha=0.2,
                )

        ax.set_xlabel(r"Iteration", fontsize=12)
        ax.set_ylabel(r"Avg. loss", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot accuracy convergence (if available)
        if has_accuracy:
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
                        and "_step" in history.columns
                        and "accuracy" in history.columns
                    ):
                        all_histories.append(history[["_step", "accuracy"]])

                # Average across trials
                if all_histories:
                    combined_history = pd.concat(all_histories)
                    step_data = combined_history.groupby("_step")["accuracy"].agg(
                        ["mean", "std"]
                    )
                    ax.plot(
                        step_data.index,
                        step_data["mean"],
                        label=format_optimizer_name(optimizer),
                        linewidth=2,
                    )
                    # Add confidence interval
                    ax.fill_between(
                        step_data.index,
                        step_data["mean"] - step_data["std"],
                        step_data["mean"] + step_data["std"],
                        alpha=0.2,
                    )

            ax.set_xlabel(r"Iteration", fontsize=12)
            ax.set_ylabel(r"Avg. accuracy (\%)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            filename = f"convergence_embd{n_embd}_head{n_head}_layer{n_layer}_gpt.pdf"
            filepath = output_dir / filename
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()


def create_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    """Create comparison plots across model sizes."""
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

    # Plot 1: Final Loss vs Model Size
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
        filepath = output_dir / "loss_vs_model_size_gpt.pdf"
        plt.savefig(filepath, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    else:
        plt.show()

    plt.close()

    # Plot 2: Final Accuracy vs Model Size (if accuracy data exists)
    if agg_df["final_accuracy_mean"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))

        for optimizer in sorted(agg_df["optimizer"].unique()):
            opt_df = agg_df[agg_df["optimizer"] == optimizer].sort_values("total_params")
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

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / "accuracy_vs_model_size_gpt.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    # Plot 3: Heatmap of loss by embedding dimension and number of layers
    # (fixing n_head to most common value for simplicity)
    most_common_n_head = df["n_head"].mode()[0]

    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[(df["optimizer"] == optimizer) & (df["n_head"] == most_common_n_head)]

        if len(opt_df) == 0:
            continue

        # Create pivot table for loss
        pivot = opt_df.pivot_table(
            values="final_loss", index="n_layer", columns="n_embd", aggfunc="mean"
        )

        if pivot.empty:
            continue

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
            f"{format_optimizer_name(optimizer)} (n\_head={most_common_n_head})",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel(r"Embedding dimension", fontsize=12)
        ax.set_ylabel(r"Number of layers", fontsize=12)

        plt.tight_layout()

        if output_dir:
            filepath = output_dir / f"heatmap_loss_{optimizer}_gpt.pdf"
            plt.savefig(filepath, bbox_inches="tight")
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    # Plot 4: Heatmap of accuracy by embedding dimension and number of layers
    if df["final_accuracy"].notna().any():
        for optimizer in sorted(df["optimizer"].unique()):
            opt_df = df[(df["optimizer"] == optimizer) & (df["n_head"] == most_common_n_head)]

            if len(opt_df) == 0:
                continue

            # Create pivot table for accuracy
            pivot = opt_df.pivot_table(
                values="final_accuracy",
                index="n_layer",
                columns="n_embd",
                aggfunc="mean",
            )

            if pivot.empty:
                continue

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
                f"{format_optimizer_name(optimizer)} (n\_head={most_common_n_head})",
                fontsize=14,
                fontweight="bold"
            )
            ax.set_xlabel(r"Embedding dimension", fontsize=12)
            ax.set_ylabel(r"Number of layers", fontsize=12)

            plt.tight_layout()

            if output_dir:
                filepath = output_dir / f"heatmap_accuracy_{optimizer}_gpt.pdf"
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
    print_comparison_by_architecture(df)
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
