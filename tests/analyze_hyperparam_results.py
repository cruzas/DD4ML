#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import analysis_helper as helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# FILENAME & ARCHITECTURE MAPPINGS
# =============================================================================
SUFFIX_MAP = {"medium_ffnn": "fcnn", "medium_cnn": "cnn", "nanogpt": "gpt"}
LATEX_OUT_DIR = Path.home() / "Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures_tex"


def write_latex_figure(filename, caption, label, extra_images=None, side_by_side=False):
    """
    Generates a LaTeX figure fragment.
    Ensures the overparameterization_legend.pdf takes full width on top
    for loss_vs_size and accuracy_vs_size plots.
    """
    LATEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = LATEX_OUT_DIR / f"{Path(filename).stem}.tex"

    safe_caption = caption.replace("%", r"\%").replace("_", r"\_")

    content = ["\\begin{figure}[htbp]", "    \\centering"]

    # Check if this is a size-related plot to include the global legend
    is_size_plot = "loss_vs_size" in filename or "accuracy_vs_size" in filename

    if is_size_plot:
        # Legend takes full width on its own line
        content.append(
            f"    \\includegraphics[width=\\linewidth]{{figures/overparameterization_legend.pdf}}\\\\"
        )

    if side_by_side and extra_images:
        # Side-by-side layout for the data plots
        content.append(
            f"    \\includegraphics[width=0.49\\linewidth]{{figures/{filename}}}"
        )
        for f in extra_images:
            content.append(
                f"    \\includegraphics[width=0.49\\linewidth]{{figures/{f}}}"
            )
    elif extra_images:
        # Vertical stacking for heatmaps
        content.append(
            f"    \\includegraphics[width=\\linewidth]{{figures/{filename}}}"
        )
        for f in extra_images:
            content.append(f"    \\includegraphics[width=\\linewidth]{{figures/{f}}}")
    else:
        # Single plot (e.g., GPT loss)
        content.append(
            f"    \\includegraphics[width=\\linewidth]{{figures/{filename}}}"
        )

    content.append(f"    \\caption{{{safe_caption}}}")
    content.append(f"    \\label{{{label}}}")
    content.append("\\end{figure}")

    with open(tex_path, "w") as f:
        f.write("\n".join(content))


def create_global_legend(output_dir):
    """Creates a standalone legend file with no border and no background."""
    helper.setup_plotting_style()
    # Use facecolor='none' to ensure the initial figure is transparent
    fig, ax = plt.subplots(figsize=(10, 1), facecolor="none")

    ax.plot(
        [],
        [],
        label=helper.format_optimizer_name("sgd"),
        marker="o",
        color=helper.COLOURS["modernBlue"],
    )
    ax.plot(
        [],
        [],
        label=helper.format_optimizer_name("apts_d", 2),
        marker="s",
        color=helper.COLOURS["modernPink"],
    )
    ax.plot(
        [],
        [],
        label=helper.format_optimizer_name("apts_d", 4),
        marker="s",
        color=helper.COLOURS["modernTeal"],
    )

    # 1. Force the legend frame to be fully transparent and have no width
    legend = ax.legend(loc="center", ncol=3, frameon=False)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_facecolor("none")

    ax.axis("off")

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    legend_path = output_dir / "overparameterization_legend.pdf"

    # 2. Use transparent=True and remove facecolor='white'
    # 3. Add pad_inches=0 to ensure the tight crop doesn't create a border artifact
    fig.savefig(legend_path, bbox_inches=bbox, transparent=True, pad_inches=0)
    plt.close(fig)


def extract_data(runs, model_type):
    """Extract and calculate parameters for specific model architecture."""
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
            "dataset": config.get(
                "dataset", "MNIST" if "gpt" not in model_type else "TinyShakespeare"
            ),
        }
        for key in arch_keys:
            row[key] = config.get(key)
        if model_type == "nanogpt":
            row["n_head"] = config.get("n_head")
        results.append(row)

    return pd.DataFrame(results).dropna(subset=["optimizer"] + arch_keys)


def create_sgd_parameter_comparison_plots(df, model_type, output_dir):
    """Generates scatter plots comparing SGD and bundles Loss/Accuracy side-by-side."""
    sgd_df = df[df["optimizer"] == "sgd"].copy()
    if sgd_df.empty:
        return

    params = sgd_df[["overlap", "batch_inc_factor"]].drop_duplicates().dropna()
    param_list = sorted(params.values.tolist(), key=lambda x: x[0])

    if len(param_list) < 2:
        return

    model_suffix = SUFFIX_MAP.get(model_type, model_type)
    p1, p2 = param_list[0], param_list[1]
    # Clean up labels for the plot axes
    c1_lab = "Config. 1"
    c2_lab = "Config. 2"

    # Detail configurations for the caption
    c1_desc = f"no overlap and no batch size increase ({c1_lab.lower()})"
    c2_desc = rf"an overlap of approximately {p2[0]*100:.0f}% and a batch size increase factor ({p2[1]:.1f}) ({c2_lab.lower()})"

    arch_keys = helper.get_arch_keys(model_type)

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

    metrics = ["loss"] if model_type == "nanogpt" else ["loss", "accuracy"]
    num_metrics = len(metrics)

    fig, axes = plt.subplots(
        1, num_metrics, figsize=(8 * num_metrics, 8), squeeze=False
    )

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        col = f"final_{metric}"
        m1, m2 = f"{col}_1", f"{col}_2"

        ax.scatter(
            merged[m1], merged[m2], s=200, alpha=0.7, edgecolors="black", zorder=3
        )

        min_val = min(merged[m1].min(), merged[m2].min())
        max_val = max(merged[m1].max(), merged[m2].max())
        lims = [min_val * 0.95, max_val * 1.05]

        ax.plot(lims, lims, "k--", alpha=0.5, zorder=2)
        ax.set_title(metric.title())
        ax.set_xlabel(c1_lab)
        ax.set_ylabel(c2_lab)

    fname = f"sgd_parameter_scatter_combined_{model_suffix}.pdf"
    helper.finalize_plot(plt.gca(), output_dir, fname)

    # Updated caption with the requested descriptions
    if "ffnn" in model_type:
        model_type_desc = "FCNNs"
    elif "cnn" in model_type:
        model_type_desc = "CNNs"
    else:
        model_type_desc = "transformers"

    cap = f"Scatter plot comparing SGD with {c1_desc} against SGD with {c2_desc} for {model_type_desc} on the {df['dataset'].iloc[0]} dataset."

    write_latex_figure(fname, cap, f"fig:SGD_to_overlap_or_not_{model_suffix}")


def create_heatmaps_for_model(df, model_type, args):
    """Generates heatmaps and bundles Loss/Accuracy vertically for FFNN/CNN."""
    arch_keys = helper.get_arch_keys(model_type)
    model_suffix = SUFFIX_MAP.get(model_type, model_type)
    metrics = ["loss", "accuracy"] if model_type != "nanogpt" else ["loss"]
    generated_files = {}

    for metric in metrics:
        labels = helper.get_arch_labels(model_type)
        heatmap_list = []

        sgd_subset = df[
            (df["optimizer"] == "sgd")
            & (np.isclose(df["overlap"], args.sgd_overlap, atol=1e-2))
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
                        values=f"final_{metric}",
                        aggfunc="mean",
                    ),
                }
            )

        sapts_df = df[
            df["optimizer"].str.contains("apts")
            & np.isclose(df["overlap"], 0.33, atol=1e-2)
        ]
        for n_sub in sorted(sapts_df["num_subdomains"].dropna().unique()):
            sub_df = sapts_df[sapts_df["num_subdomains"] == n_sub]
            if not sub_df.empty:
                heatmap_list.append(
                    {
                        "title": helper.format_optimizer_name("apts_d", int(n_sub)),
                        "data": sub_df.pivot_table(
                            index=arch_keys[1],
                            columns=arch_keys[0],
                            values=f"final_{metric}",
                            aggfunc="mean",
                        ),
                    }
                )

        if heatmap_list:
            n_plots = len(heatmap_list)
            fig, axes = plt.subplots(
                1, n_plots, figsize=(7 * n_plots, 6), squeeze=False
            )
            all_vals = pd.concat([h["data"].stack() for h in heatmap_list])
            vmin, vmax = all_vals.min(), all_vals.max()
            cmap = helper.get_custom_heat_cmap(metric)

            for i, h in enumerate(heatmap_list):
                sns.heatmap(
                    h["data"],
                    annot=True,
                    fmt=".4f" if metric == "loss" else ".2f",
                    ax=axes[0, i],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    cbar=(i == n_plots - 1),
                )
                axes[0, i].set_title(h["title"])
                axes[0, i].set_xlabel(labels[0])
                axes[0, i].set_ylabel(labels[1] if i == 0 else "")

            fname = f"heatmap_{metric}_combined_{model_suffix}.pdf"
            helper.finalize_plot(plt.gca(), args.output_dir, fname)
            generated_files[metric] = fname

    if model_type == "nanogpt":
        cap = f"Heatmaps showing the final average empirical loss for SAPTS-D and SGD on transformers on the TinyShakespeare dataset."
        write_latex_figure(generated_files["loss"], cap, f"fig:heatmap_gpt_sgd_vs_apts")
    else:
        if model_type == "medium_ffnn":
            cap = f"Heatmaps showing the final average empirical loss (top) and final average test accuracy (bottom) for SAPTS-D and SGD on {model_suffix.upper()}s of varying depths and widths on the MNIST dataset."
        else:
            cap = f"Heatmaps showing the final average empirical loss (top) and final average test accuracy (bottom) for SAPTS-D and SGD on {model_suffix.upper()}s of varying depths and filter sizes on the MNIST dataset."
        write_latex_figure(
            generated_files["loss"],
            cap,
            f"fig:loss_acc_heatmap_{model_suffix}",
            extra_images=[generated_files["accuracy"]],
        )


def run_analysis(df, model_type, args):
    """Core analysis loop per model type."""
    model_suffix = SUFFIX_MAP.get(model_type, model_type)
    helper.setup_plotting_style()

    # 1. Performance vs Size Line Plots (Side-by-Side LaTeX bundling)
    size_files = {}
    for metric in ["loss", "accuracy"]:
        col = f"final_{metric}"
        if col not in df.columns or df[col].isna().all():
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        for opt, mark, lw in [("sgd", "o", 4), ("apts", "s", 3)]:
            subset = df[df["optimizer"].str.contains(opt)]
            if opt == "sgd":
                subset = subset[
                    np.isclose(subset["overlap"], args.sgd_overlap, atol=1e-2)
                ]

            for n in (
                [None]
                if opt == "sgd"
                else sorted(subset["num_subdomains"].dropna().unique())
            ):
                data = subset if n is None else subset[subset["num_subdomains"] == n]
                if data.empty:
                    continue
                agg = (
                    data.groupby("total_params")[col].agg(["mean", "std"]).reset_index()
                )
                ax.errorbar(
                    agg["total_params"],
                    agg["mean"],
                    yerr=agg["std"],
                    marker=mark,
                    linewidth=lw,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Total Parameters")
        if metric.title().lower() == "loss":
            ax.set_ylabel(f"Final Empirical {metric.title()}")
        else:
            ax.set_ylabel(f"Final Test {metric.title()}")
        fname = f"{metric}_vs_size_{model_suffix}.pdf"
        helper.finalize_plot(ax, args.output_dir, fname)
        size_files[metric] = fname

    if model_suffix == "gpt":
        cap = (
            f"Final average loss plotted against parameter count for transformers. "
            "SAPTS-D consistently achieves lower loss compared to SGD "
            "across all tested architectures on the TinyShakespeare dataset. Error bars represent $\\pm 1$ standard deviation."
        )
    else:
        cap = (
            f"Final average loss (left) and test accuracy (right) plotted against parameter count for {model_suffix.upper()}s. "
            "SAPTS-D consistently achieves lower loss and higher accuracy compared to SGD "
            "across all tested architectures on the MNIST dataset. Error bars represent $\\pm 1$ standard deviation."
        )
    if "loss" in size_files and "accuracy" in size_files:
        write_latex_figure(
            size_files["loss"],
            cap,
            f"fig:loss_acc_{model_suffix}_sgd_vs_apts",
            extra_images=[size_files["accuracy"]],
            side_by_side=True,
        )
    elif "loss" in size_files:
        write_latex_figure(
            size_files["loss"], cap, f"fig:loss_acc_{model_suffix}_sgd_vs_apts"
        )

    # 2. Combined Heatmaps
    create_heatmaps_for_model(df, model_type, args)

    # 3. SGD Ablation Comparison Scatter (Side-by-Side LaTeX bundling inside function)
    create_sgd_parameter_comparison_plots(df, model_type, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ohtests")
    parser.add_argument(
        "--entity", type=str, default="cruzas-universit-della-svizzera-italiana"
    )
    parser.add_argument("--save-plots", action="store_true")
    # Removed --sgd-overlap from argparse
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["medium_ffnn", "medium_cnn", "nanogpt"],
    )
    args = parser.parse_args()

    if args.save_plots:
        legend_dir = Path(__file__).parent / "figures"
        legend_dir.mkdir(exist_ok=True)
        create_global_legend(legend_dir)

    for model_type in args.models:
        # Determine overlap based on the model type
        if model_type == "nanogpt":
            current_sgd_overlap = 0.33
        else:
            current_sgd_overlap = 0.0

        # Attach it to args so run_analysis and create_heatmaps_for_model can still access it
        args.sgd_overlap = current_sgd_overlap

        runs = helper.fetch_runs(
            project=args.project,
            entity=args.entity,
            filters={"config.model_name": model_type},
            model_type=model_type,
        )

        if not runs:
            print(f"No runs found for {model_type}")
            continue

        df = extract_data(runs, model_type)
        if df.empty:
            continue

        args.output_dir = (
            Path(__file__).parent
            / f"hyperparam_analysis_{SUFFIX_MAP.get(model_type, model_type)}"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Analyzing {model_type} with SGD overlap set to {args.sgd_overlap}...")
        run_analysis(df, model_type, args)


if __name__ == "__main__":
    main()
