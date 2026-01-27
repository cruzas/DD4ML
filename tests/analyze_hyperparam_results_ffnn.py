#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import analysis_helper as helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

helper.setup_plotting_style()


def extract_run_data(runs: List) -> pd.DataFrame:
    results = []
    print(f"\nProcessing {len(runs)} runs...")

    for run in runs:
        config = run.config
        summary = run.summary._json_dict

        width = config.get("width", None)
        num_layers = config.get("num_layers", None)

        try:
            width = int(width) if width is not None else None
        except (ValueError, TypeError):
            width = None

        try:
            num_layers = int(num_layers) if num_layers is not None else None
        except (ValueError, TypeError):
            num_layers = None

        if width and num_layers:
            total_params = (
                (784 * width) + ((num_layers - 1) * width * width) + (width * 10)
            )
        else:
            total_params = None

        optimizer = config.get("optimizer", "unknown")
        trial = config.get("trial", 1)
        epochs = config.get("epochs", None)
        batch_size = config.get("batch_size", None)
        learning_rate = config.get("learning_rate", None)
        num_subdomains = config.get("num_subdomains", None)
        overlap = config.get("overlap", None)
        batch_inc_factor = config.get("batch_inc_factor", None)

        final_loss = summary.get("loss", None)
        final_accuracy = summary.get("accuracy", None)
        total_runtime = summary.get("running_time", None)
        total_grad_evals = summary.get("grad_evals", None)

        history = helper.load_history_cached(run.id, run)

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
    initial_count = len(df)
    df = df.dropna(subset=["optimizer", "width", "num_layers"])
    filtered_count = initial_count - len(df)

    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} runs missing critical config info")

    print(f"  Successfully processed {len(df)} runs")
    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Optimizers tested: {sorted(df['optimizer'].unique())}")
    print(f"Network widths: {sorted(df['width'].unique())}")
    print(f"Network depths: {sorted(df['num_layers'].unique())}")
    print(f"Trials per configuration: {df['trial'].max()}")

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
    print("\n" + "=" * 80)
    print("COMPARISON BY NETWORK SIZE")
    print("=" * 80)
    architectures = (
        df.groupby(["width", "num_layers"])
        .size()
        .reset_index()[["width", "num_layers"]]
    )

    for _, arch in architectures.iterrows():
        width, num_layers = arch["width"], arch["num_layers"]
        arch_df = df[(df["width"] == width) & (df["num_layers"] == num_layers)]
        if len(arch_df) == 0:
            continue

        print(f"\n{'─' * 80}")
        print(
            f"Width={width}, Depth={num_layers} ({arch_df['total_params'].iloc[0]:,} parameters)"
        )
        print(f"{'─' * 80}")
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
    print("\n" + "=" * 80)
    print("HYPERPARAMETERIZATION EFFECT ANALYSIS")
    print("=" * 80)

    for optimizer in sorted(df["optimizer"].unique()):
        opt_df = df[df["optimizer"] == optimizer]
        print(f"\n{optimizer.upper()}:")
        print(f"{'─' * 60}")
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
            width, depth = int(row["width"]), int(row["num_layers"])
            l_m, l_s = row[("final_loss", "mean")], row[("final_loss", "std")]
            a_m, a_s = row[("final_accuracy", "mean")], row[("final_accuracy", "std")]
            print(
                f"{params:<12} {width:<8} {depth:<8} {l_m:.6f}±{l_s:.6f}    {a_m:.4f}±{a_s:.4f}"
            )


def create_convergence_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    print("\n" + "=" * 80)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 80)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    architectures = (
        df.groupby(["width", "num_layers"])
        .size()
        .reset_index()[["width", "num_layers"]]
    )

    for _, arch in architectures.iterrows():
        width, num_layers = arch["width"], arch["num_layers"]
        arch_df = df[(df["width"] == width) & (df["num_layers"] == num_layers)]
        if len(arch_df) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Width={width}, Depth={num_layers}", fontweight="bold")

        for i, metric in enumerate(["loss", "accuracy"]):
            ax = axes[i]
            for optimizer in sorted(arch_df["optimizer"].unique()):
                all_histories = []
                for _, run_data in arch_df[
                    arch_df["optimizer"] == optimizer
                ].iterrows():
                    h = run_data["history"]
                    if (
                        h is not None
                        and not h.empty
                        and "epoch" in h.columns
                        and metric in h.columns
                    ):
                        all_histories.append(h[["epoch", metric]])
                if all_histories:
                    comb = (
                        pd.concat(all_histories)
                        .groupby("epoch")[metric]
                        .agg(["mean", "std"])
                    )
                    ax.plot(
                        comb.index,
                        comb["mean"],
                        label=helper.format_optimizer_name(optimizer),
                        linewidth=2,
                    )
                    ax.fill_between(
                        comb.index,
                        comb["mean"] - comb["std"],
                        comb["mean"] + comb["std"],
                        alpha=0.2,
                    )
            ax.set_xlabel(r"Epoch")
            ax.set_ylabel(f"Avg. {metric}")
            ax.legend()

        plt.tight_layout()
        if output_dir:
            plt.savefig(
                output_dir / f"convergence_w{width}_nl{num_layers}_ffnn.pdf",
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()


def create_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None, sgd_overlap: float = 0.0
) -> None:
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    sns.set_style("whitegrid")
    param_combinations = []
    all_params = df[["overlap", "batch_inc_factor"]].drop_duplicates()
    for _, row in all_params.iterrows():
        if pd.notna(row["overlap"]) and pd.notna(row["batch_inc_factor"]):
            param_combinations.append((row["overlap"], row["batch_inc_factor"]))

    if not param_combinations:
        param_combinations = [(None, None)]

    for ov_val, bif_val in param_combinations:
        if ov_val is not None:
            apts_subset = df[
                (df["optimizer"].isin(["apts_d", "apts_p", "apts_ip"]))
                & (abs(df["overlap"] - ov_val) < 0.01)
                & (abs(df["batch_inc_factor"] - bif_val) < 0.01)
            ]
            sgd_subset = df[
                (df["optimizer"] == "sgd") & (abs(df["overlap"] - sgd_overlap) < 0.01)
            ]
            df_subset = pd.concat([sgd_subset, apts_subset])
            suffix = f"_overlap{ov_val:.2f}_batchinc{bif_val:.2f}".replace(".", "_")
        else:
            df_subset, suffix = df, ""

        if len(df_subset) == 0:
            continue

        cols = ["optimizer", "total_params"]
        if any(
            opt in df_subset["optimizer"].unique()
            for opt in ["apts_d", "apts_p", "apts_ip"]
        ):
            cols.append("num_subdomains")

        agg_df = (
            df_subset.groupby(cols)
            .agg(
                {
                    "final_loss": ["mean", "std"],
                    "final_accuracy": ["mean", "std"],
                    "total_runtime": ["mean", "std"],
                }
            )
            .reset_index()
        )
        agg_df.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in agg_df.columns.values
        ]

        for metric in ["loss", "accuracy"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for opt in sorted(agg_df["optimizer"].unique()):
                if (
                    opt in ["apts_d", "apts_p", "apts_ip"]
                    and "num_subdomains" in agg_df.columns
                ):
                    for n in sorted(
                        agg_df[agg_df["optimizer"] == opt]["num_subdomains"]
                        .dropna()
                        .unique()
                    ):
                        sub = agg_df[
                            (agg_df["optimizer"] == opt)
                            & (agg_df["num_subdomains"] == n)
                        ].sort_values("total_params")
                        ax.errorbar(
                            sub["total_params"],
                            sub[f"final_{metric}_mean"],
                            yerr=sub[f"final_{metric}_std"],
                            marker="o",
                            capsize=5,
                            label=helper.format_optimizer_name(opt, int(n)),
                        )
                else:
                    sub = agg_df[agg_df["optimizer"] == opt].sort_values("total_params")
                    ax.errorbar(
                        sub["total_params"],
                        sub[f"final_{metric}_mean"],
                        yerr=sub[f"final_{metric}_std"],
                        marker="o",
                        capsize=5,
                        label=helper.format_optimizer_name(opt),
                    )

            ax.set_xscale("log")
            ax.set_ylabel(f"Final avg. {metric}")
            ax.legend()
            if ov_val is not None:
                txt = f"SAPTS: ov={ov_val:.2f}, bif={bif_val:.2f}\nSGD: ov={sgd_overlap:.2f}"
                ax.text(
                    0.98,
                    0.98,
                    txt,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        facecolor=helper.COLOURS["background"],
                        edgecolor=helper.COLOURS["modernDark"],
                        alpha=0.8,
                    ),
                )

            plt.tight_layout()
            if output_dir:
                plt.savefig(output_dir / f"{metric}_vs_network_size_ffnn{suffix}.pdf")
            else:
                plt.show()
            plt.close()

    # Heatmap logic
    sgd_selected = df[
        (df["optimizer"] == "sgd") & (abs(df["overlap"] - sgd_overlap) < 0.01)
    ]
    heat_opts = ["apts_d", "apts_p", "apts_ip"]

    for metric in ["loss", "accuracy"]:
        heatmap_data = []
        global_vals = []

        if not sgd_selected.empty:
            p = sgd_selected.pivot_table(
                values=f"final_{metric}",
                index="num_layers",
                columns="width",
                aggfunc="mean",
            )
            if not p.empty:
                heatmap_data.append(
                    {"pivot": p, "title": helper.format_optimizer_name("sgd")}
                )
                global_vals.extend(p.values.flatten())

        for opt in heat_opts:
            opt_df = df[
                (df["optimizer"] == opt)
                & (abs(df["overlap"] - 0.33) < 0.01)
                & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
            ]
            for n in sorted(opt_df["num_subdomains"].dropna().unique()):
                p = opt_df[opt_df["num_subdomains"] == n].pivot_table(
                    values=f"final_{metric}",
                    index="num_layers",
                    columns="width",
                    aggfunc="mean",
                )
                if not p.empty:
                    heatmap_data.append(
                        {"pivot": p, "title": helper.format_optimizer_name(opt, int(n))}
                    )
                    global_vals.extend(p.values.flatten())

        if heatmap_data:
            vmin, vmax = min(global_vals), max(global_vals)
            n_heatmaps = len(heatmap_data)
            n_cols = min(3, n_heatmaps)
            n_rows = (n_heatmaps + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes = np.array(axes).flatten()

            for idx, data in enumerate(heatmap_data):
                sns.heatmap(
                    data["pivot"],
                    annot=True,
                    fmt=".4f" if metric == "loss" else ".2f",
                    cmap=helper.get_custom_heat_cmap(metric),
                    ax=axes[idx],
                    vmin=vmin,
                    vmax=vmax,
                    cbar=False,
                )
                axes[idx].set_title(data["title"], fontweight="bold")

            for idx in range(n_heatmaps, len(axes)):
                axes[idx].axis("off")

            from matplotlib import cm
            from matplotlib.colors import Normalize

            sm = cm.ScalarMappable(
                cmap=helper.get_custom_heat_cmap(metric),
                norm=Normalize(vmin=vmin, vmax=vmax),
            )
            plt.tight_layout(rect=[0, 0, 0.95, 1])
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            fig.colorbar(sm, cax=cbar_ax).set_label(f"Final avg. {metric}")

            if output_dir:
                plt.savefig(output_dir / f"heatmap_{metric}_combined_ffnn.pdf")
            else:
                plt.show()
            plt.close()


def create_sgd_parameter_comparison_plots(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> None:
    print("\n" + "=" * 80)
    print("GENERATING SGD PARAMETER COMPARISON PLOTS")
    print("=" * 80)
    sgd_df = df[df["optimizer"] == "sgd"].copy()
    if len(sgd_df) == 0:
        return

    params = sgd_df[["overlap", "batch_inc_factor"]].drop_duplicates().dropna()
    param_list = sorted(params.values.tolist(), key=lambda x: x[0])
    if len(param_list) < 2:
        return

    # Helper to generate descriptive labels for the annotation
    def get_config_label(overlap, batch_inc):
        if abs(overlap) < 0.01 and abs(batch_inc - 1.0) < 0.01:
            return "no overlap \\& no batch increase"
        elif abs(overlap - 0.33) < 0.01 and abs(batch_inc - 1.5) < 0.01:
            return "overlap \\& batch increase"
        else:
            return f"ov={overlap*100:.0f}%, bif={batch_inc:.2f}"

    overlap1, bif1 = param_list[0]
    overlap2, bif2 = param_list[1]
    config1_label = get_config_label(overlap1, bif1)
    config2_label = get_config_label(overlap2, bif2)

    for metric in ["loss", "accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for ov, bif in param_list:
            subset = sgd_df[
                (abs(sgd_df["overlap"] - ov) < 0.01)
                & (abs(sgd_df["batch_inc_factor"] - bif) < 0.01)
            ]
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
                label=f"ov={ov*100:.0f}%, bif={bif:.2f}",
            )
        ax.set_xscale("log")
        ax.set_ylabel(f"Final avg. {metric}")
        ax.set_xlabel("Number of parameters")
        ax.legend()
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f"sgd_parameter_comparison_{metric}_ffnn.pdf")
        plt.close()

    # Scatter plot logic
    sgd1 = sgd_df[
        (abs(sgd_df["overlap"] - overlap1) < 0.01)
        & (abs(sgd_df["batch_inc_factor"] - bif1) < 0.01)
    ]
    sgd2 = sgd_df[
        (abs(sgd_df["overlap"] - overlap2) < 0.01)
        & (abs(sgd_df["batch_inc_factor"] - bif2) < 0.01)
    ]

    m = pd.merge(
        sgd1.groupby(["width", "num_layers"])
        .agg({"final_loss": "mean", "final_accuracy": "mean"})
        .reset_index(),
        sgd2.groupby(["width", "num_layers"])
        .agg({"final_loss": "mean", "final_accuracy": "mean"})
        .reset_index(),
        on=["width", "num_layers"],
        suffixes=("_1", "_2"),
    )

    for metric in ["loss", "accuracy"]:
        m1, m2 = f"final_{metric}_1", f"final_{metric}_2"
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(m[m1], m[m2], s=100, alpha=0.6, edgecolors="black")

        # Diagonal line
        lims = [min(m[m1].min(), m[m2].min()), max(m[m1].max(), m[m2].max())]
        ax.plot(lims, lims, "k--", alpha=0.5)

        # Determine "better" logic based on metric type
        if metric == "loss":
            # Lower loss is better
            better_text = f"Below diagonal = {config2_label} better\nAbove diagonal = {config1_label} better"
        else:
            # Higher accuracy is better
            better_text = f"Above diagonal = {config2_label} better\nBelow diagonal = {config1_label} better"

        ax.set_xlabel(f"Final {metric}: {config1_label}")
        ax.set_ylabel(f"Final {metric}: {config2_label}")

        ax.text(
            0.05,
            0.95,
            better_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                facecolor=helper.COLOURS["background"], alpha=0.5, boxstyle="round"
            ),
        )

        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f"sgd_parameter_scatter_{metric}_ffnn.pdf")
        else:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameterization test results from wandb"
    )
    parser.add_argument("--project", type=str, default="ohtests")
    parser.add_argument(
        "--entity", type=str, default="cruzas-universit-della-svizzera-italiana"
    )
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./hyperparam_analysis")
    )
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--filter-optimizer", type=str, nargs="+", default=None)
    parser.add_argument("--filter-model", type=str, default="medium_ffnn")
    parser.add_argument("--no-cache", action="store_false")
    parser.add_argument("--cache-max-age", type=int, default=24)
    parser.add_argument("--sgd-overlap", type=float, default=0.0)
    args = parser.parse_args()

    if args.save_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    filters = {}
    if args.filter_optimizer:
        filters["config.optimizer"] = {"$in": args.filter_optimizer}
    if args.filter_model:
        filters["config.model_name"] = args.filter_model

    runs = helper.fetch_runs(
        project=args.project,
        entity=args.entity,
        filters=filters if filters else None,
        use_cache=not args.no_cache,
        cache_max_age_hours=args.cache_max_age,
    )
    if not runs:
        return

    df = extract_run_data(runs)
    if df.empty:
        return

    helper.validate_apts_parameters(df)
    print_summary_statistics(df)
    print_comparison_by_network_size(df)
    print_hyperparameterization_analysis(df)

    out = args.output_dir if args.save_plots else None
    try:
        create_comparison_plots(df, out, sgd_overlap=args.sgd_overlap)
        create_sgd_parameter_comparison_plots(df, out)
    except Exception as e:
        print(f"\nWarning: Error creating plots: {e}")

    if args.export_csv:
        helper.export_results(df, args.output_dir / "hyperparam_results.csv")

    print("\n" + "=" * 80 + "\nANALYSIS COMPLETE\n" + "=" * 80)


if __name__ == "__main__":
    main()
