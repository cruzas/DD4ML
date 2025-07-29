#!/usr/bin/env python3
"""
Enhanced time_series_grid.py (no interpolation)

Plots time-series grids by using raw history data (no interpolation) for both epoch and running_time.
Saves one CSV per optimizer/num_stages combo, pivoted with columns for each metric.
"""
import math
import os
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    csv_path: str | None = None,
):
    """
    Fetch runs, compute means & stds per raw x-axis values (no interpolation), then plot time-series grid.
    Saves one CSV per optimizer/num_stages combo if csv_path is given.

    CSV for each combo is pivoted with columns:
      - x_axis,
      - <metric>,
      - <metric>_std
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
                "config.num_stages": 1,
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
            (r.config.get("optimizer", "unk"), r.config.get("num_stages", "unk"))
            for r in runs_all
        },
        key=lambda x: (str(x[0]), x[1]),
    )
    labels_full = [rf"${latex_opt(opt.upper())}\;\mid\;N={N}$" for opt, N in combos]
    colour_map = {combo: f"C{i}" for i, combo in enumerate(combos)}

    # Prepare storage for CSV rows per combo
    csv_rows_by_combo: dict[tuple, list[dict]] = {combo: [] for combo in combos}

    # Organise runs by (size, combo)
    runs_by = {}
    for r in runs_all:
        bs = r.config.get(base_key)
        combo = (r.config.get("optimizer", "unk"), r.config.get("num_stages", "unk"))
        runs_by.setdefault((bs, combo), []).append(r)

    # Create subplot grid
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(batch_sizes),
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    xlabel_map = {"epoch": "epoch", "running_time": "time (s)"}

    for i, metric in enumerate(metrics):
        for j, size in enumerate(batch_sizes):
            ax = axes[i, j]
            for combo in combos:
                # collect raw series
                series_list = []
                for r in runs_by.get((size, combo), []):
                    hist = get_history(r.id)
                    if x_axis in hist.columns and metric in hist.columns:
                        df = hist[[x_axis, metric]].dropna()
                        series_list.append(df)
                if not series_list:
                    continue

                # concatenate on x_axis index
                df_concat = pd.concat(
                    [s.set_index(x_axis)[metric] for s in series_list], axis=1
                )
                # compute mean & std across runs for each x-axis value
                mean_series = df_concat.mean(axis=1)
                std_series = df_concat.std(axis=1)

                # collect CSV rows
                for x_val, m, s in zip(
                    mean_series.index, mean_series.values, std_series.values
                ):
                    csv_rows_by_combo[combo].append(
                        {
                            base_key: size,
                            "metric": metric,
                            "optimizer": combo[0],
                            "num_stages": combo[1],
                            x_axis: x_val,
                            "mean": m,
                            "std": s,
                        }
                    )

                # plot
                col = colour_map[combo]
                ax.plot(mean_series.index, mean_series.values, color=col)
                if len(series_list) > 1:
                    ax.fill_between(
                        mean_series.index,
                        mean_series.values - std_series.values,
                        mean_series.values + std_series.values,
                        alpha=0.2,
                        color=col,
                    )

            prefix = "BS" if base_key == "batch_size" else "EBS"
            if i == 0:
                ax.set_title(f"{prefix}={size}")
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel_map.get(x_axis, x_axis))
            ax.grid(True, alpha=0.3)
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
        lg = plt.figure(
            figsize=(max(6, ncols * 1.2), 1.5 * math.ceil(len(labels_full) / ncols))
        )
        lg.legend(legend_handles, labels_full, loc="center", ncol=ncols, frameon=False)
        lg_path = os.path.join(os.path.dirname(save_path), f"{dataset}_legend.pdf")
        lg.savefig(lg_path, bbox_inches="tight")
        plt.close(lg)
    else:
        plt.show()

    # Write one pivoted CSV per combo
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        for combo, rows in csv_rows_by_combo.items():
            opt, stages = combo
            df = pd.DataFrame(rows)
            means = df.pivot(index=x_axis, columns="metric", values="mean")
            stds = df.pivot(index=x_axis, columns="metric", values="std")
            pivoted = means.join(stds.add_suffix("_std")).reset_index()
            cols = [x_axis]
            for m in metrics:
                cols.append(m)
                cols.append(f"{m}_std")
            pivoted = pivoted[cols]

            base, _ = os.path.splitext(csv_path)
            out_path = f"{base}_{opt}_N{stages}.csv"
            pivoted.to_csv(out_path, index=False)
            print(f"Exported pivoted CSV for {opt}, N={stages} to {out_path}")


def main():
    entity = "cruzas-universit-della-svizzera-italiana"
    project = "gamm2025"
    proj = f"{entity}/{project}"
    base_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")

    datasets = ["cifar10"]
    model_names = {"mnist": "medium_ffnn", "cifar10": "big_resnet"}
    configs = {
        "mnist": {
            "filters": {"config.model_name": model_names["mnist"]},
            "sgd_filters_strong": {10000: {"learning_rate": 0.10}},
            "x_limits": {"epoch": (0, 100), "running_time": (0, 300)},
            "y_limits": {"loss": (0, 2.0), "accuracy": (20, 100)},
        },
        "cifar10": {
            "filters": {"config.model_name": model_names["cifar10"]},
            "sgd_filters_strong": {200: {"learning_rate": 0.1}},
            "x_limits": {"epoch": (0, 25), "running_time": (0, 800)},
            "y_limits": {"loss": (0, 5.0), "accuracy": (20, 90)},
        },
    }
    x_axes = ["epoch", "running_time"]

    for dataset in datasets:
        cfg = configs[dataset]
        regimes = {
            "strong": {
                "sizes": sorted(cfg["sgd_filters_strong"].keys()),
                "base_key": "batch_size",
                "sgd_key": "sgd_filters_strong",
            }
        }
        fb = {"config.dataset_name": dataset, **cfg["filters"]}
        metrics = ["loss", "accuracy"]
        for regime, params in regimes.items():
            for x_axis in x_axes:
                fname = f"{dataset}_{model_names[dataset]}_{regime}_{x_axis}_grid.pdf"
                save_path = os.path.join(base_dir, fname)
                csv_path = save_path.replace("pdf", "csv")
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
                    csv_path=csv_path,
                )


if __name__ == "__main__":
    main()
