#!/usr/bin/env python3
import argparse
from pathlib import Path

import analysis_helper as helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_data(runs, model_type):
    results = []
    arch_keys = helper.get_arch_keys(model_type)

    for run in runs:
        config = run.config
        summary = run.summary._json_dict

        row = {
            "run_id": run.id,
            "optimizer": config.get("optimizer", "unknown"),
            "total_params": helper.calculate_params(model_type, config),
            "trial": config.get("trial", 1),
            "overlap": config.get("overlap", 0.0),
            "batch_inc_factor": config.get("batch_inc_factor", 1.0),
            "num_subdomains": config.get("num_subdomains"),
            "final_loss": summary.get("loss"),
            "final_accuracy": summary.get("accuracy"),
            "runtime": summary.get("running_time"),
        }
        # Add architecture specific keys (e.g., width, filters)
        for key in arch_keys:
            row[key] = config.get(key)

        # For GPT models, we also need n_head to filter properly later
        if model_type == "nanogpt":
            row["n_head"] = config.get("n_head")

        results.append(row)

    return pd.DataFrame(results).dropna(subset=["optimizer"] + arch_keys)


def create_heatmaps(df, model_type, arch_keys, metric, sgd_overlap, output_dir):
    """Generates a combined heatmap for SGD (filtered) and SAPTS variants."""
    labels = helper.get_arch_labels(model_type)

    # 1. Prepare Heatmap Data
    heatmap_list = []

    # Filter SGD Baseline based on user preference
    sgd_subset = df[
        (df["optimizer"] == "sgd") & (np.isclose(df["overlap"], sgd_overlap, atol=1e-2))
    ]

    if model_type == "nanogpt":
        n_head = helper.get_common_n_head(df)
        sgd_subset = sgd_subset[sgd_subset["n_head"] == n_head]

    if not sgd_subset.empty:
        heatmap_list.append(
            {
                "title": helper.format_optimizer_name("sgd"),
                "data": sgd_subset.pivot_table(
                    index=arch_keys[1],
                    columns=arch_keys[0],
                    values=metric,
                    aggfunc="mean",
                ),
            }
        )

    # Add SAPTS variants (usually filtered by standard overlap=0.33, batch_inc=1.5)
    sapts_df = df[
        df["optimizer"].str.contains("apts")
        & np.isclose(df["overlap"], 0.33, atol=1e-2)
        & np.isclose(df["batch_inc_factor"], 1.5, atol=1e-2)
    ]

    for n_sub in sorted(sapts_df["num_subdomains"].dropna().unique()):
        sub_df = sapts_df[sapts_df["num_subdomains"] == n_sub]
        if model_type == "nanogpt":
            sub_df = sub_df[sub_df["n_head"] == n_head]

        if not sub_df.empty:
            heatmap_list.append(
                {
                    "title": helper.format_optimizer_name("apts_d", int(n_sub)),
                    "data": sub_df.pivot_table(
                        index=arch_keys[1],
                        columns=arch_keys[0],
                        values=metric,
                        aggfunc="mean",
                    ),
                }
            )

    if not heatmap_list:
        return

    # 2. Plotting
    n_plots = len(heatmap_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

    all_vals = pd.concat([h["data"].stack() for h in heatmap_list])
    vmin, vmax = all_vals.min(), all_vals.max()
    cmap = helper.get_custom_heat_cmap("loss" if "loss" in metric else "accuracy")

    for i, h in enumerate(heatmap_list):
        sns.heatmap(
            h["data"],
            annot=True,
            fmt=".4f" if "loss" in metric else ".2f",
            ax=axes[0, i],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar=(i == n_plots - 1),
        )
        axes[0, i].set_title(h["title"])
        axes[0, i].set_xlabel(labels[0])
        axes[0, i].set_ylabel(labels[1] if i == 0 else "")

    plt.tight_layout()
    helper.finalize_plot(plt.gca(), output_dir, f"heatmap_combined_{metric}.pdf")


def run_analysis(df, model_type, args):
    output_dir = args.output_dir
    arch_keys = helper.get_arch_keys(model_type)
    sns.set_style("whitegrid")

    # 1. Line Plots: Performance vs Model Size
    for metric in ["final_loss", "final_accuracy"]:
        if df[metric].isna().all():
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot SGD Baseline
        sgd_line = df[
            (df["optimizer"] == "sgd")
            & (np.isclose(df["overlap"], args.sgd_overlap, atol=1e-2))
        ]
        if not sgd_line.empty:
            agg = (
                sgd_line.groupby("total_params")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                agg["total_params"],
                agg["mean"],
                yerr=agg["std"],
                label=helper.format_optimizer_name("sgd") + f" (ov={args.sgd_overlap})",
                marker="o",
                capsize=5,
            )

        # Plot SAPTS series for each N
        sapts_data = df[df["optimizer"].str.contains("apts")]
        for n_sub in sorted(sapts_data["num_subdomains"].dropna().unique()):
            subset = sapts_data[sapts_data["num_subdomains"] == n_sub].sort_values(
                "total_params"
            )
            if not subset.empty:
                agg = (
                    subset.groupby("total_params")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                ax.errorbar(
                    agg["total_params"],
                    agg["mean"],
                    yerr=agg["std"],
                    label=helper.format_optimizer_name("apts_d", int(n_sub)),
                    marker="s",
                    capsize=5,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Total Parameters")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        helper.finalize_plot(ax, output_dir, f"line_{metric}_vs_size.pdf")

    # 2. Combined Heatmaps
    for metric in ["final_loss", "final_accuracy"]:
        if not df[metric].isna().all():
            create_heatmaps(
                df, model_type, arch_keys, metric, args.sgd_overlap, output_dir
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ohtests")
    parser.add_argument(
        "--entity", type=str, default="cruzas-universit-della-svizzera-italiana"
    )
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--sgd-overlap",
        type=float,
        default=0.0,
        help="Overlap to use for SGD baseline (0.0 or 0.33)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["medium_ffnn", "medium_cnn", "nanogpt"],
    )
    args = parser.parse_args()

    dir_map = {
        "medium_ffnn": "hyperparam_analysis_ffnn",
        "medium_cnn": "hyperparam_analysis_cnn",
        "nanogpt": "hyperparam_analysis_gpt",
    }

    for model_type in args.models:
        folder_name = dir_map.get(model_type, f"hyperparam_analysis_{model_type}")
        model_output_dir = Path(__file__).parent / folder_name
        if args.save_plots:
            model_output_dir.mkdir(parents=True, exist_ok=True)

        runs = helper.fetch_runs(
            project=args.project,
            entity=args.entity,
            filters={"config.model_name": model_type},
            model_type=model_type,
        )
        if not runs:
            continue

        df = extract_data(runs, model_type)
        if df.empty:
            continue

        args.output_dir = model_output_dir
        print(f"\n>>> Analyzing {model_type} (SGD Overlap: {args.sgd_overlap})...")
        run_analysis(df, model_type, args)


if __name__ == "__main__":
    main()
