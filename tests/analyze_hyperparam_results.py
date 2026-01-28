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
        for key in arch_keys:
            row[key] = config.get(key)

        if model_type == "nanogpt":
            row["n_head"] = config.get("n_head")

        results.append(row)

    return pd.DataFrame(results).dropna(subset=["optimizer"] + arch_keys)


def create_sgd_parameter_comparison_plots(df, model_type, output_dir):
    """Pairwise comparison of SGD with different overlap/batch_inc settings."""
    sgd_df = df[df["optimizer"] == "sgd"].copy()
    if sgd_df.empty:
        return

    # Identify unique param sets
    params = sgd_df[["overlap", "batch_inc_factor"]].drop_duplicates().dropna()
    param_list = sorted(params.values.tolist(), key=lambda x: x[0])

    if len(param_list) < 2:
        print(f"  Skipping SGD comparison for {model_type}: < 2 parameter sets found.")
        return

    arch_keys = helper.get_arch_keys(model_type)

    # Configuration Labels
    p1, p2 = param_list[0], param_list[1]
    c1_lab = f"ov={p1[0]*100:.0f}%, bif={p1[1]:.2f}"
    c2_lab = f"ov={p2[0]*100:.0f}%, bif={p2[1]:.2f}"

    # 1. Line Plot Comparison (Loss and Accuracy)
    for metric in ["final_loss", "final_accuracy"]:
        if sgd_df[metric].isna().all():
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        for ov, bif in param_list:
            subset = sgd_df[
                (np.isclose(sgd_df["overlap"], ov, atol=1e-2))
                & (np.isclose(sgd_df["batch_inc_factor"], bif, atol=1e-2))
            ]
            agg = (
                subset.groupby("total_params")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                agg["total_params"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=f"ov={ov*100:.0f}%, bif={bif:.2f}",
            )

        ax.set_xscale("log")
        ax.set_title(f"SGD: {model_type} Parameter Comparison")
        ax.set_xlabel("Total Parameters")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        helper.finalize_plot(ax, output_dir, f"sgd_line_comp_{metric}.pdf")

    # 2. Scatter Plot (Direct Mapping)
    sgd1 = (
        sgd_df[(np.isclose(sgd_df["overlap"], p1[0], atol=1e-2))]
        .groupby(arch_keys)
        .agg({"final_loss": "mean", "final_accuracy": "mean"})
        .reset_index()
    )
    sgd2 = (
        sgd_df[(np.isclose(sgd_df["overlap"], p2[0], atol=1e-2))]
        .groupby(arch_keys)
        .agg({"final_loss": "mean", "final_accuracy": "mean"})
        .reset_index()
    )

    merged = pd.merge(sgd1, sgd2, on=arch_keys, suffixes=("_1", "_2"))

    if not merged.empty:
        for metric in ["final_loss", "final_accuracy"]:
            m1, m2 = f"{metric}_1", f"{metric}_2"
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(
                merged[m1], merged[m2], s=180, alpha=0.7, edgecolors="black", zorder=3
            )

            # Diagonal line
            lims = [
                min(merged[m1].min(), merged[m2].min()) * 0.95,
                max(merged[m1].max(), merged[m2].max()) * 1.05,
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, zorder=2)

            ax.set_xlabel(f"Config 1: {c1_lab}")
            ax.set_ylabel(f"Config 2: {c2_lab}")
            ax.set_title(f"SGD {metric.split('_')[-1].title()} Comparison")

            # Logic for "which side is better"
            if "loss" in metric:
                better_text = (
                    f"Above diagonal: Favours Config 1 ({c1_lab})\n"
                    f"Below diagonal: Favours Config 2 ({c2_lab})"
                )
            else:
                better_text = (
                    f"Above diagonal: Favours Config 2 ({c2_lab})\n"
                    f"Below diagonal: Favours Config 1 ({c1_lab})"
                )

            ax.text(
                0.05,
                0.95,
                better_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=16,
                bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
            )

            helper.finalize_plot(ax, output_dir, f"sgd_scatter_{metric}.pdf")


def create_heatmaps(df, model_type, arch_keys, metric, sgd_overlap, output_dir):
    labels = helper.get_arch_labels(model_type)
    heatmap_list = []

    # Filter SGD
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

    # Filter SAPTS
    sapts_df = df[
        df["optimizer"].str.contains("apts")
        & np.isclose(df["overlap"], 0.33, atol=1e-2)
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

    n_plots = len(heatmap_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6), squeeze=False)

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
            annot_kws={"size": 16},
        )  # Larger font inside heatmap
        axes[0, i].set_title(h["title"])
        axes[0, i].set_xlabel(labels[0])
        axes[0, i].set_ylabel(labels[1] if i == 0 else "")

    plt.tight_layout()
    helper.finalize_plot(plt.gca(), output_dir, f"heatmap_combined_{metric}.pdf")


def run_analysis(df, model_type, args):
    output_dir = args.output_dir
    arch_keys = helper.get_arch_keys(model_type)
    helper.setup_plotting_style()  # Ensure global style is applied

    # 1. Performance vs Size Line Plots
    for metric in ["final_loss", "final_accuracy"]:
        if df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(12, 8))

        # SGD Baseline
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
                capsize=6,
                linewidth=4,
            )

        # SAPTS Series
        sapts_data = df[df["optimizer"].str.contains("apts")]
        for n_sub in sorted(sapts_data["num_subdomains"].dropna().unique()):
            subset = sapts_data[sapts_data["num_subdomains"] == n_sub]
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
                    capsize=6,
                    linewidth=3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Total Parameters")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(loc="best", frameon=True)
        helper.finalize_plot(ax, output_dir, f"line_{metric}_vs_size.pdf")

    # 2. Combined Heatmaps
    for metric in ["final_loss", "final_accuracy"]:
        if not df[metric].isna().all():
            create_heatmaps(
                df, model_type, arch_keys, metric, args.sgd_overlap, output_dir
            )

    # 3. SGD Parameter Comparison (New)
    create_sgd_parameter_comparison_plots(df, model_type, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ohtests")
    parser.add_argument(
        "--entity", type=str, default="cruzas-universit-della-svizzera-italiana"
    )
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--sgd-overlap", type=float, default=0.0)
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
