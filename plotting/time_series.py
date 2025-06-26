import os
import pprint
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import wandb
from plotting.utils import _abbreviate_val


def plot_time_series(
    project_path: str,
    x_axis: str,
    metric: str,
    filters: dict = None,
    group_by: list[str] = None,
    group_by_abbr: dict[str, str] = None,
    show_variance: bool = True,
    figsize: tuple = (8, 6),
    save_path: str = None,
):
    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    print(f"Found {len(runs)} runs for filters {filters!r}")

    if runs:
        print("Columns in first run.history():", runs[0].history().columns.tolist())

    # prepare grouping
    group_by = group_by or []
    abbr = {k: k for k in group_by}
    if group_by_abbr:
        abbr.update(group_by_abbr)
    grouped = {}
    for run in runs:
        if group_by:
            key = tuple(run.config.get(k, "unk") for k in group_by)
            lbl = " | ".join(f"{abbr[k]}={v}" for k, v in zip(group_by, key))
        else:
            key, lbl = "all", "all runs"
        grouped.setdefault(key, {"runs": [], "label": lbl})["runs"].append(run)

    # set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {x_axis}")
    ax.grid(True, alpha=0.3)

    for info in grouped.values():
        series = []
        for run in info["runs"]:
            hist = run.history()
            if x_axis in hist.columns and metric in hist.columns:
                series.append(hist[[x_axis, metric]].dropna())
        if not series:
            continue

        # common grid + interpolation
        all_x = np.concatenate([s[x_axis].values for s in series])
        grid = np.linspace(all_x.min(), all_x.max(), 200)
        data = np.stack([np.interp(grid, s[x_axis], s[metric]) for s in series])
        m, s_ = data.mean(0), data.std(0)

        ax.plot(grid, m, label=f"{info['label']} (n={len(series)})")
        if show_variance and len(series) > 1:
            ax.fill_between(grid, m - s_, m + s_, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
):
    proj = f"{entity}/{project}"
    base_save = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    datasets = ["mnist"]
    plots = [
        # ("epoch", "loss"),
        # ("epoch", "accuracy"),
        ("grad_evals", "loss"),
        ("grad_evals", "accuracy"),
        ("running_time", "loss"),
        ("running_time", "accuracy"),
    ]

    regimes = {
        "strong": {
            "sizes": [1024, 2048, 4096],
            "base_key": "batch_size",
            "group_by": ["optimizer", "batch_size", "num_subdomains"],
            "abbr": {"optimizer": "opt", "batch_size": "bs", "num_subdomains": "N"},
        },
        "weak": {
            "sizes": [128, 256, 512],
            "base_key": "effective_batch_size",
            "group_by": ["optimizer", "effective_batch_size", "num_subdomains"],
            "abbr": {
                "optimizer": "opt",
                "effective_batch_size": "effbs",
                "num_subdomains": "N",
            },
        },
    }

    for regime, cfg in regimes.items():
        for dataset, size in product(datasets, cfg["sizes"]):
            filters = {
                "config.dataset_name": dataset,
                f"config.{cfg['base_key']}": size,
            }
            for x_axis, metric in plots:
                fname = f"{dataset}_{regime}_bs_{size}_" f"{metric}_vs_{x_axis}.pdf"
                save_path = os.path.join(base_save, fname)
                plot_time_series(
                    project_path=proj,
                    x_axis=x_axis,
                    metric=metric,
                    filters=filters,
                    group_by=cfg["group_by"],
                    group_by_abbr=cfg["abbr"],
                    save_path=save_path,
                )


if __name__ == "__main__":
    main()
