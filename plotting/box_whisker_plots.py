import os
import pprint
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from plotting.analysis import analyze_wandb_runs_advanced


def plot_box_whisker(
    grouped_df,
    metrics,
    show_variance=True,
    plot_type="bar",
    aggregate="mean",
    save_path="~/OneDrive/Documents/PhD/thesis_plots",
):
    """Plot grouped metrics with optional variance—uses `group_label` for ticks"""
    if grouped_df is None or grouped_df.empty:
        pprint.pprint("No grouped data to plot")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for i, m in enumerate(metrics):
        mean_c = f"summary_{m}_{aggregate}"
        std_c = f"summary_{m}_std"
        cnt_c = f"summary_{m}_count"
        if mean_c not in grouped_df:
            pprint.pprint(f"Column {mean_c} not found")
            continue

        labels = grouped_df["group_label"]
        vals = grouped_df[mean_c]

        if plot_type == "bar":
            axes[i].bar(range(len(labels)), vals, alpha=0.7)
            if show_variance and std_c in grouped_df:
                errs = grouped_df[std_c] / np.sqrt(grouped_df[cnt_c])
                axes[i].errorbar(
                    range(len(labels)), vals, yerr=errs, fmt="none", capsize=5
                )
        else:  # scatter
            axes[i].scatter(range(len(labels)), vals, s=100, alpha=0.7)
            if show_variance and std_c in grouped_df:
                errs = grouped_df[std_c] / np.sqrt(grouped_df[cnt_c])
                axes[i].errorbar(
                    range(len(labels)), vals, yerr=errs, fmt="none", capsize=5
                )

        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45, ha="right")
        axes[i].set_title(f"{m}")
        axes[i].set_ylabel(f"{m} ({aggregate})")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
    # plt.show()


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="tr_variants_assessment",
):
    base_save = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    datasets = ["mnist"]
    optimizers = ["lssr1_tr", "tr"]
    batch_sizes = [15000, 30000, 60000]
    group_by_map = {"glob_second_order": "so", "glob_dogleg": "dleg"}

    experiments = [
        (
            {
                "config.optimizer": opt,
                "config.model_name": "simple_ffnn",
                "config.dataset_name": ds,
                "config.batch_size": bs,
            },
            group_by_map,
            os.path.join(base_save, f"{opt}_{ds}_{bs}"),
        )
        for ds, opt, bs in product(datasets, optimizers, batch_sizes)
    ]

    pprint.pprint(experiments)

    for i, (filters, group_by_map, save_path) in enumerate(experiments, start=1):
        pprint.pprint(f"\n=== Experiment {i} ===")
        group_by = list(group_by_map.keys())
        group_by_abbr = list(group_by_map.values())
        metrics = ["accuracy", "loss"]

        df, gdf = analyze_wandb_runs_advanced(
            f"{entity}/{project}",
            filters=filters,
            group_by=group_by,
            group_by_abbr=group_by_abbr,
            metrics=metrics,
            show_variance=True,
            aggregate="mean",
            mad_threshold=3,
        )

        if gdf is not None:
            pprint.pprint(
                gdf[
                    [
                        "group_label",
                        "summary_loss_mean",
                        "summary_loss_std",
                        "summary_loss_count",
                    ]
                ]
            )
            plot_box_whisker(
                gdf,
                metrics,
                show_variance=True,
                plot_type="bar",
                save_path=save_path,
            )


if __name__ == "__main__":
    main(
        entity="cruzas-universit-della-svizzera-italiana",
        project="tr_variants_assessment",
    )
