#!/usr/bin/env python3
"""
Enhanced time_series_grid.py

Plots time-series grids with dataset-specific filters and SGD learning-rate overrides,
for strong and weak scaling across mnist, cifar10, tinyshakespeare, and poisson2d.
Allows specification of per-dataset axis limits for x-axis (per x_axis), y-axis (per metric),
and logarithmic scales.
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
        return rf"\\mathrm{{{parts[0]}}}_{{\\mathrm{{{parts[1]}}}}}"
    return rf"\\mathrm{{{opt}}}"


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
    x_scales: dict[str, str] | None = None,
    y_scales: dict[str, str] | None = None,
    save_path: str | None = None,
):
    """
    Fetch runs with general + optional per-size SGD filters, then plot time-series grid.
    x_limits: optional dict mapping x_axis names to (min, max).
    y_limits: optional dict mapping metric names to (min, max).
    x_scales, y_scales: optional dict mapping axis names to 'linear' or 'log'.
    """
    extra_filters = extra_filters or {}
    sgd_filters = sgd_filters or {}
    api = wandb.Api()

    # Collect runs
    runs_all = []
    for size in batch_sizes:
        base_f = {**filters_base, f"config.{base_key}": size, **extra_filters}
        runs = api.runs(project_path, filters=base_f)
        non_sgd = [
            r
            for r in runs
            if r.config.get(base_key) == size
            and r.config.get("optimizer", "").lower() != "sgd"
        ]
        runs_all.extend(non_sgd)
        if size in sgd_filters:
            lr_cfg = sgd_filters[size]
            sgd_f = {
                **base_f,
                "config.optimizer": "sgd",
                "config.num_subdomains": 1,
                **{f"config.{k}": v for k, v in lr_cfg.items()},
            }
            sgd_runs = api.runs(project_path, filters=sgd_f)
            runs_all.extend([r for r in sgd_runs if r.config.get(base_key) == size])

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
    labels = [rf"${latex_opt(opt.upper())}\;\mid\;N={N}$" for opt, N in combos]
    colour_map = {c: f"C{i}" for i, c in enumerate(combos)}

    # Group runs
    runs_by = {}
    for r in runs_all:
        combo = (
            r.config.get("optimizer", "unk"),
            r.config.get("num_subdomains", "unk"),
        )
        runs_by.setdefault((r.config.get(base_key), combo), []).append(r)

    # Create grid
    fig, axes = plt.subplots(
        len(metrics), len(batch_sizes), figsize=figsize, sharex=True, squeeze=False
    )
    xlabel = {"grad_evals": r"$\#\mathrm{grad}$", "running_time": "time (s)"}

    for i, metric in enumerate(metrics):
        for j, size in enumerate(batch_sizes):
            ax = axes[i, j]
            for combo in combos:
                dfs = []
                for r in runs_by.get((size, combo), []):
                    hist = get_history(r.id)
                    if x_axis in hist.columns and metric in hist.columns:
                        dfs.append(hist[[x_axis, metric]].dropna())
                if not dfs:
                    continue
                xs = np.concatenate([d[x_axis].values for d in dfs])
                grid = np.linspace(xs.min(), xs.max(), 200)
                arr = np.stack([np.interp(grid, d[x_axis], d[metric]) for d in dfs])
                m, s = arr.mean(axis=0), arr.std(axis=0)
                col = colour_map[combo]
                ax.plot(grid, m, color=col)
                if len(dfs) > 1:
                    ax.fill_between(grid, m - s, m + s, alpha=0.2, color=col)

            if i == 0:
                ax.set_title(f"{'BS' if base_key=='batch_size' else 'EBS'}={size}")
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel.get(x_axis, x_axis))
            ax.grid(True, alpha=0.3)

            # limits
            if x_limits and x_axis in x_limits:
                ax.set_xlim(*x_limits[x_axis])
            if y_limits and metric in y_limits:
                ax.set_ylim(*y_limits[metric])
            # scales
            if x_scales and x_axis in x_scales:
                ax.set_xscale(x_scales[x_axis])
            if y_scales and metric in y_scales:
                ax.set_yscale(y_scales[metric])

    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        # legend
        lg = plt.figure(
            figsize=(max(6, len(labels) // 4 * 1.2), 1.5 * math.ceil(len(labels) / 4))
        )
        handles = [Line2D([0], [0], color=colour_map[c], lw=2) for c in combos]
        lg.legend(
            handles, labels, loc="center", ncol=min(len(labels), 4), frameon=False
        )
        lg.savefig(
            os.path.join(
                os.path.dirname(save_path),
                f"{filters_base.get('config.dataset_name')}_legend.pdf",
            ),
            bbox_inches="tight",
        )
        plt.close(lg)
    else:
        plt.show()


def main():
    entity = "cruzas-universit-della-svizzera-italiana"
    project = "thesis_results"
    proj = f"{entity}/{project}"
    base_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")

    datasets = ["mnist", "cifar10", "tinyshakespeare", "poisson2d"]
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
            "x_limits": {"grad_evals": (0, 1000), "running_time": (0, 500)},
            "y_limits": {"loss": (0.0, 100.0)},
        },
    }

    x_axes = ["grad_evals", "running_time"]
    for dataset in datasets:
        cfg = configs[dataset]
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
            y_scales_map = {"loss": "log"}
        elif dataset == "tinyshakespeare":
            metrics = ["loss", "train_perplexity"]
            y_scales_map = None
        else:
            metrics = ["loss", "accuracy"]
            y_scales_map = None

        for regime, params in regimes.items():
            for x_axis in x_axes:
                fname = f"{dataset}_{regime}_{x_axis}_grid.pdf"
                spath = os.path.join(base_dir, fname)
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
                    x_scales=None,
                    y_scales=y_scales_map,
                    save_path=spath,
                )


if __name__ == "__main__":
    main()
