import hashlib
import os
import pprint
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced

# ——— Cache setup ———————————————————————————————————————————
CACHE_DIR = os.path.expanduser("~/.cache/wandb_analysis")
os.makedirs(CACHE_DIR, exist_ok=True)


def _make_key(params: dict) -> str:
    """Generate a deterministic hash key from parameters."""
    serialized = repr(sorted(params.items())).encode("utf-8")
    return hashlib.md5(serialized).hexdigest()


def analyze_cached(
    entity: str,
    project: str,
    *,
    filters: dict,
    group_by: list,
    group_by_abbr: list,
    metrics: list,
    show_variance: bool,
    aggregate: str,
    mad_threshold: float,
    recompute: bool = False,
):
    """Wrap analyze_wandb_runs_advanced with on-disk caching of grouped_df only."""
    params = {
        "entity": entity,
        "project": project,
        **{f"filter:{k}": v for k, v in sorted(filters.items())},
        "group_by": tuple(group_by),
        "group_by_abbr": tuple(group_by_abbr),
        "metrics": tuple(metrics),
        "show_variance": show_variance,
        "aggregate": aggregate,
        "mad_threshold": mad_threshold,
    }
    key = _make_key(params)
    cache_path = os.path.join(CACHE_DIR, f"{key}_gdf.pkl")

    if not recompute and os.path.exists(cache_path):
        print(f"Loading cached grouped_df from {cache_path}")
        gdf = pd.read_pickle(cache_path)
        df = None
    else:
        df, gdf = analyze_wandb_runs_advanced(
            f"{entity}/{project}",
            filters=filters,
            group_by=group_by,
            group_by_abbr=group_by_abbr,
            metrics=metrics,
            show_variance=show_variance,
            aggregate=aggregate,
            mad_threshold=mad_threshold,
        )
        pd.to_pickle(gdf, cache_path)

    return df, gdf


# ——— Plotting function —————————————————————————————————————
def plot_box_whisker(
    grouped_df,
    metrics,
    show_variance=True,
    plot_type="bar",
    aggregate="mean",
    save_path="~/OneDrive/Documents/PhD/thesis_plots",
):
    """Plot grouped metrics with optional variance—uses `group_label` for ticks."""
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
        else:
            axes[i].scatter(range(len(labels)), vals, s=100, alpha=0.7)

        if show_variance and std_c in grouped_df:
            errs = grouped_df[std_c] / np.sqrt(grouped_df[cnt_c])
            axes[i].errorbar(range(len(labels)), vals, yerr=errs, fmt="none", capsize=5)

        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45, ha="right")
        axes[i].set_title(m)
        axes[i].set_ylabel(f"{m} ({aggregate})")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
    # plt.show()


# ——— Main execution ———————————————————————————————————————
def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="tr_variants_assessment",
    recompute=True,
):
    base_save = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    datasets = ["mnist"]
    optimizers = ["tr"]
    batch_sizes = [15000]
    paper_tr_update = False
    group_by_map = {
        "glob_second_order": "so",
        "glob_dogleg": "dleg",
        "paper_tr_update": "ptru",
    }

    experiments = [
        (
            {
                "config.optimizer": opt,
                "config.model_name": "simple_ffnn",
                "config.dataset_name": ds,
                "config.batch_size": bs,
            },
            group_by_map,
            os.path.join(base_save, f"{opt}_{ds}_{bs}.pdf"),
        )
        for ds, opt, bs in product(datasets, optimizers, batch_sizes)
    ]

    for filters, group_by_map, save_path in experiments:
        pprint.pprint(f"\n=== Experiment {save_path} ===")
        txt_path = os.path.splitext(save_path)[0] + ".txt"

        group_by = list(group_by_map.keys())
        group_by_abbr = list(group_by_map.values())
        metrics = ["accuracy", "loss"]

        df, gdf = analyze_cached(
            entity=entity,
            project=project,
            filters=filters,
            group_by=group_by,
            group_by_abbr=group_by_abbr,
            metrics=metrics,
            show_variance=True,
            aggregate="mean",
            mad_threshold=3,
            recompute=recompute,
        )

        if gdf is not None:
            subset = gdf[
                [
                    "group_label",
                    "summary_loss_mean",
                    "summary_loss_std",
                    "summary_loss_count",
                    "summary_accuracy_mean",
                    "summary_accuracy_std",
                    "summary_accuracy_count",
                ]
            ]

            with open(txt_path, "w") as f:
                f.write(pprint.pformat(subset.to_dict(orient="records")))

            plot_box_whisker(
                gdf,
                metrics,
                show_variance=True,
                plot_type="bar",
                save_path=save_path,
            )


if __name__ == "__main__":
    main()
