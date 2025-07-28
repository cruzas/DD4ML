#!/usr/bin/env python3
"""
Enhanced time_series_grid.py

Plots time-series grids with dataset-specific filters and SGD learning-rate overrides,
for strong and weak scaling across mnist, cifar10, and tinyshakespeare.
Allows specification of per-dataset axis limits for x-axis (per x_axis) and y-axis (per metric).
"""
import math
import os
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import wandb

# Matplotlib LaTeX and style settings
mpl.rcParams.update(
    {
        "text.usetex": True,
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


def latex_opt(opt: str) -> str:
    """Format optimizer name for LaTeX rendering."""
    parts = opt.split("_", 1)
    if len(parts) == 2:
        return rf"\mathrm{{{parts[0]}}}_{{\mathrm{{{parts[1]}}}}}"
    return rf"\mathrm{{{opt}}}"


def plot_grid_time_series(
    project_path: str,
    x_axis: str,
    base_key: str,
    batch_sizes: list[int],
    metrics: list[str],
    filters_base: dict,
    extra_filters: dict | None = None,
    sgd_filters: dict[int, dict] | None = None,
    figsize: tuple = (12, 8),
    x_limits: dict[str, tuple[float, float]] | None = None,
    y_limits: dict[str, tuple[float, float]] | None = None,
    save_path: str | None = None,
    y_log: bool = False,
):
    """
    Fetch runs with general + optional per-size SGD filters, then plot time-series grid.
    x_limits: optional dict mapping x_axis names to (min, max) tuples.
    y_limits: optional dict mapping metric names to (min, max) tuples.
    """
    extra_filters = extra_filters or {}
    sgd_filters = sgd_filters or {}
    api = wandb.Api()

    # Collect runs
    runs_all = []
    for size in batch_sizes:
        base_f = {**filters_base, f"config.{base_key}": size, **extra_filters}
        runs_gen = api.runs(project_path, filters=base_f)
        runs_gen = [
            r
            for r in runs_gen
            if r.config.get(base_key) == size
            and r.config.get("optimizer", "").lower() != "sgd"
        ]
        runs_all.extend(runs_gen)
        if size in sgd_filters:
            lr_f = sgd_filters[size]
            sgd_f = {
                **base_f,
                "config.optimizer": "sgd",
                "config.num_subdomains": 1,
                **{f"config.{k}": v for k, v in lr_f.items()},
            }
            runs_sgd = api.runs(project_path, filters=sgd_f)
            runs_sgd = [r for r in runs_sgd if r.config.get(base_key) == size]
            runs_all.extend(runs_sgd)

    print(f"Collected total of {len(runs_all)} runs.")
    run_by_id = {r.id: r for r in runs_all}

    @lru_cache(maxsize=None)
    def get_history(run_id: str):
        return run_by_id[run_id].history()

    combos = sorted(
        {
            (r.config.get("optimizer", "unk"), r.config.get("num_subdomains", "unk"))
            for r in runs_all
        },
        key=lambda x: (str(x[0]), x[1]),
    )
    labels_full = [rf"${latex_opt(opt.upper())}\;\mid\;N={N}$" for opt, N in combos]
    colour_map = {combo: f"C{i}" for i, combo in enumerate(combos)}

    # Organize runs by (size, combo)
    runs_by = {}
    for r in runs_all:
        bs = r.config.get(base_key)
        combo = (
            r.config.get("optimizer", "unk"),
            r.config.get("num_subdomains", "unk"),
        )
        runs_by.setdefault((bs, combo), []).append(r)

    # Create subplot grid
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(batch_sizes),
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    xlabel_map = {"grad_evals": r"$\#\mathrm{grad}$", "running_time": "time (s)"}

    for i, metric in enumerate(metrics):
        for j, size in enumerate(batch_sizes):
            ax = axes[i, j]
            for combo in combos:
                series_list = []
                for r in runs_by.get((size, combo), []):
                    hist = get_history(r.id)
                    if x_axis in hist.columns and metric in hist.columns:
                        df = hist[[x_axis, metric]].dropna()
                        series_list.append(df)
                if not series_list:
                    continue
                all_x = np.concatenate([s[x_axis].values for s in series_list])
                grid = np.linspace(all_x.min(), all_x.max(), 200)
                data = np.stack(
                    [np.interp(grid, s[x_axis], s[metric]) for s in series_list]
                )
                mean = data.mean(axis=0)
                std = data.std(axis=0)
                color = colour_map[combo]
                ax.plot(grid, mean, color=color)
                if len(series_list) > 1:
                    ax.fill_between(
                        grid, mean - std, mean + std, alpha=0.2, color=color
                    )

            prefix = "BS" if base_key == "batch_size" else "EBS"
            if i == 0:
                ax.set_title(f"{prefix}={size}")
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel_map.get(x_axis, x_axis))
            ax.grid(True, alpha=0.3)

            # Apply per-axis limits if provided
            if x_limits and x_axis in x_limits:
                ax.set_xlim(*x_limits[x_axis])
            if y_limits and metric in y_limits:
                ax.set_ylim(*y_limits[metric])
            if y_log:
                ax.set_yscale("log")

    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Save figure and legend
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        dataset = filters_base.get("config.dataset_name", "dataset")
        legend_handles = [Line2D([0], [0], color=colour_map[c], lw=2) for c in combos]
        ncols = min(len(labels_full), 4)
        legend_fig = plt.figure(
            figsize=(max(6, ncols * 1.2), 1.5 * math.ceil(len(labels_full) / ncols))
        )
        legend_fig.legend(
            legend_handles, labels_full, loc="center", ncol=ncols, frameon=False
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

    datasets = ["poisson2d"]
    configs = {
        "mnist": {
            "filters": {"config.model_name": "simple_cnn"},
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.10},
                2048: {"learning_rate": 0.10},
                4096: {"learning_rate": 0.10},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.01},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.10},
            },
            "x_limits": {"grad_evals": (0, 40), "running_time": (0, 250)},
            "y_limits": {"loss": (0, 2.0), "accuracy": (20, 100)},
        },
        "cifar10": {
            "filters": {"config.model_name": "simple_resnet"},
            "sgd_filters_strong": {
                2048: {"learning_rate": 0.1},
                4096: {"learning_rate": 0.1},
                8192: {"learning_rate": 0.1},
            },
            "sgd_filters_weak": {
                256: {"learning_rate": 0.01},
                512: {"learning_rate": 0.1},
                1024: {"learning_rate": 0.1},
            },
            "x_limits": {"grad_evals": (0, 40), "running_time": (0, 2000)},
            "y_limits": {"loss": (0, 5.0), "accuracy": (20, 100)},
        },
        "tinyshakespeare": {
            "filters": {"config.model_name": "minigpt"},
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.01},
                2048: {"learning_rate": 0.01},
                4096: {"learning_rate": 0.01},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.10},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.10},
            },
            "x_limits": {"grad_evals": (0, 5), "running_time": (0, 500)},
            "y_limits": {"loss": (2.0, 3.0), "train_perplexity": (10, 20)},
        },
        "poisson2d": {
            "filters": {"config.model_name": "pinn_ffnn"},
            "sgd_filters_strong": {
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.001},
            },
            "sgd_filters_weak": {
                64: {"learning_rate": 0.01},
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.10},
            },
            # "x_limits": {"grad_evals": (0, 10), "running_time": (0, 1000)},
            # "y_limits": {"loss": (0, 1.0), "accuracy": (20, 100)},
        },
    }

    x_axes = ["grad_evals", "running_time"]

    for dataset in datasets:
        cfg = configs[dataset]
        # build regimes from this dataset’s own sgd_filters
        regimes = {
            "strong": {
                "sizes": sorted(cfg["sgd_filters_strong"].keys()),
                "base_key": "batch_size",
                "sgd_key": "sgd_filters_strong",
            },
            "weak": {
                "sizes": sorted(cfg["sgd_filters_weak"].keys()),
                "base_key": "effective_batch_size",
                "sgd_key": "sgd_filters_weak",
            },
        }

        fb = {"config.dataset_name": dataset, **cfg["filters"]}

        if dataset == "poisson2d":
            metrics = ["loss"]
        elif dataset == "tinyshakespeare":
            metrics = ["loss", "train_perplexity"]
        else:
            metrics = ["loss", "accuracy"]

        for regime, params in regimes.items():
            for x_axis in x_axes:
                fname = f"{dataset}_{regime}_{x_axis}_grid.pdf"
                save_path = os.path.join(base_dir, fname)
                plot_grid_time_series(
                    project_path=proj,
                    x_axis=x_axis,
                    base_key=params["base_key"],
                    batch_sizes=params["sizes"],
                    metrics=metrics,
                    filters_base=fb,
                    extra_filters=None,
                    sgd_filters=cfg[params["sgd_key"]],
                    x_limits=cfg.get("x_limits"),
                    y_limits=cfg.get("y_limits"),
                    save_path=save_path,
                    y_log=(dataset == "poisson2d"),
                )


if __name__ == "__main__":
    main()
