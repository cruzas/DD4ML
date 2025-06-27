import os
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import wandb

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
    }
)


def plot_grid_time_series(
    project_path: str,
    x_axis: str,
    base_key: str,
    batch_sizes: list[int],
    metrics: list[str],
    filters_base: dict,
    group_by: list[str],
    group_by_abbr: dict[str, str],
    figsize: tuple = (12, 8),
    save_path: str = None,
):
    api = wandb.Api()

    # Fetch all runs once and apply base filters
    print(f"Fetching runs for project: {project_path} with filters: {filters_base}")
    runs_all = api.runs(project_path, filters=filters_base)
    runs_all = [
        r
        for r in runs_all
        if r.config.get(base_key) in batch_sizes
        and not (
            r.config.get("optimizer", "").lower() == "sgd"
            and r.config.get("num_subdomains", 1) != 1
        )
    ]

    print(f"Found {len(runs_all)} runs matching criteria.")

    # Cache histories by run ID
    run_by_id = {r.id: r for r in runs_all}

    @lru_cache(maxsize=None)
    def get_history(run_id: str):
        return run_by_id[run_id].history()

    # Determine colour/label combos once, and build a reusable colour map
    combos = sorted(
        {
            (r.config.get("optimizer", "unk"), r.config.get("num_subdomains", "unk"))
            for r in runs_all
        },
        key=lambda x: (str(x[0]), x[1]),
    )
    labels_full = [f"{opt} | N={N}" for opt, N in combos]
    colour_map = {combo: f"C{i}" for i, combo in enumerate(combos)}

    # Group runs by (batch_size, combo)
    runs_by_bs_and_combo: dict[tuple[int, tuple[str, int]], list] = {}
    for r in runs_all:
        bs = r.config.get(base_key)
        combo = (
            r.config.get("optimizer", "unk"),
            r.config.get("num_subdomains", "unk"),
        )
        runs_by_bs_and_combo.setdefault((bs, combo), []).append(r)

    # Prepare subplots
    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=len(batch_sizes), figsize=figsize, sharex=True
    )
    xlabel_map = {"grad_evals": "#grad", "running_time": "time (s)"}

    for i, metric in enumerate(metrics):
        for j, size in enumerate(batch_sizes):
            ax = axes[i, j]
            for combo in combos:
                series_list = []
                for r in runs_by_bs_and_combo.get((size, combo), []):
                    hist = get_history(r.id)
                    if x_axis in hist.columns and metric in hist.columns:
                        df = hist[[x_axis, metric]].dropna()
                        series_list.append(df)
                if not series_list:
                    continue

                # Interpolate onto a common grid
                all_x = np.concatenate([s[x_axis].values for s in series_list])
                grid = np.linspace(all_x.min(), all_x.max(), 200)
                data = np.stack(
                    [np.interp(grid, s[x_axis], s[metric]) for s in series_list]
                )
                m = data.mean(axis=0)
                s_ = data.std(axis=0)

                colour = colour_map[combo]
                ax.plot(grid, m, color=colour)
                if len(series_list) > 1:
                    ax.fill_between(grid, m - s_, m + s_, alpha=0.2, color=colour)

            ax.set_title(f"{metric.title()} (bs={size})")
            ax.grid(True, alpha=0.3)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel_map.get(x_axis, x_axis))
            if j == 0:
                ax.set_ylabel(metric)

    # Finalise layout
    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Save a single legend per dataset
        dataset = filters_base.get("config.dataset_name", "dataset")
        legend_handles = [
            Line2D([0], [0], color=colour_map[combo], lw=2) for combo in combos
        ]
        legend_fig = plt.figure(figsize=(max(6, len(labels_full) * 1.2), 2))
        legend_fig.legend(
            legend_handles,
            labels_full,
            loc="center",
            ncol=len(labels_full),
            frameon=False,
        )
        legend_path = os.path.join(os.path.dirname(save_path), f"{dataset}_legend.pdf")
        legend_fig.savefig(legend_path, bbox_inches="tight")
        plt.close(legend_fig)
    else:
        plt.show()


def main():
    entity = "cruzas-universit-della-svizzera-italiana"
    project = "thesis_results"
    proj = f"{entity}/{project}"
    base_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")

    datasets = ["mnist"]
    x_axes = ["grad_evals", "running_time"]
    regimes = {
        "strong": {
            "sizes": [1024, 2048, 4096],
            "base_key": "batch_size",
        },
        "weak": {
            "sizes": [128, 256, 512],
            "base_key": "effective_batch_size",
        },
    }

    for regime, cfg in regimes.items():
        for dataset in datasets:
            filters_base = {"config.dataset_name": dataset}
            if dataset == "tinyshakespeare":
                metrics = ["loss", "train_perplexity"]
            else:
                metrics = ["loss", "accuracy"]

            for x_axis in x_axes:
                fname = f"{dataset}_{regime}_{x_axis}_grid.pdf"
                save_path = os.path.join(base_dir, fname)
                plot_grid_time_series(
                    project_path=proj,
                    x_axis=x_axis,
                    base_key=cfg["base_key"],
                    batch_sizes=cfg["sizes"],
                    metrics=metrics,
                    filters_base=filters_base,
                    group_by=[],
                    group_by_abbr={},
                    save_path=save_path,
                )


if __name__ == "__main__":
    main()
