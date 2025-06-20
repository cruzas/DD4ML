import matplotlib.pyplot as plt
import numpy as np

import wandb

from .utils import _abbreviate_val


def plot_averaged_time_series(
    project_path: str,
    filters: dict = None,
    group_by: list[str] = None,
    group_by_abbr: list[str] = None,
    metrics: list[str] = ("accuracy", "loss"),
    x_axis: str = "_step",
    show_variance: bool = True,
    figsize: tuple = (14, 6),
):
    """Plot two time-series (metrics[0] and metrics[1]) vs x_axis in side-by-side subplots."""

    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    # normalize group_by arguments
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(group_by_abbr, str):
        group_by_abbr = [group_by_abbr]
    abbr_map = dict(zip(group_by or [], group_by_abbr or []))

    # group runs
    grouped = {}
    for run in runs:
        if group_by:
            key = tuple(run.config.get(k, "unknown") for k in group_by)
            label = " | ".join(
                f"{abbr_map.get(k,k)}={('T' if v is True else 'F' if v is False else v)}"
                for k, v in zip(group_by, key)
            )
        else:
            key, label = "all", "All runs"
        grouped.setdefault(key, {"runs": [], "label": label})["runs"].append(run)

    # prepare subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    for ax, metric in zip(axes, metrics):
        ax.set_xlabel(x_axis)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {x_axis}")
        ax.grid(True, alpha=0.3)

    # plot each group
    for info in grouped.values():
        label = f"{info['label']} (n={len(info['runs'])})"
        series_data = []
        for run in info["runs"]:
            hist = run.history()
            if (
                x_axis in hist.columns
                and metrics[0] in hist.columns
                and metrics[1] in hist.columns
            ):
                series_data.append(hist[[x_axis, metrics[0], metrics[1]]].dropna())

        if not series_data:
            continue

        # define common grid per group
        all_x = np.concatenate([s[x_axis].values for s in series_data])
        grid = np.linspace(all_x.min(), all_x.max(), 200)

        for i, metric in enumerate(metrics):
            data = np.stack(
                [np.interp(grid, s[x_axis], s[metric]) for s in series_data]
            )
            m = data.mean(axis=0)
            s = data.std(axis=0)

            axes[i].plot(grid, m, label=label, linewidth=2)
            if show_variance and len(series_data) > 1:
                axes[i].fill_between(grid, m - s, m + s, alpha=0.2)

    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.show()


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
            os.path.join(base_save, f"{opt}_{ds}_{bs}.pdf"),
        )
        for ds, opt, bs in product(datasets, optimizers, batch_sizes)
    ]

    for i, (filters, group_by_map, save_path) in enumerate(experiments, start=1):
        pprint.pprint(f"\n=== Experiment {save_path} ===")
        txt_path = os.path.splitext(save_path)[0] + ".txt"

        group_by = list(group_by_map.keys())
        group_by_abbr = list(group_by_map.values())
        metrics = ["accuracy", "loss"]

        # Plotting the time series
        #...

if __name__ == "__main__":
    main(
        entity="cruzas-universit-della-svizzera-italiana",
        project="tr_variants_assessment",
    )
