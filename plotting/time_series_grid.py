#!/usr/bin/env python3
"""
Enhanced time_series_grid.py

Plots time-series grids with dataset-specific filters and SGD learning-rate overrides,
for strong and weak scaling across mnist, cifar10, and tinyshakespeare.
Allows specification of per-dataset axis limits for x-axis (per x_axis) and y-axis (per metric).

Now with:
- 'Avg. <metric>' y-axis labels (LaTeX-safe for %).
- Choice of combo key: 'num_stages' or 'num_subdomains'.
- On-disk caching of per-run histories, grouped by dataset.
- On-disk caching of run listings (api.runs) with a TTL.
"""
import hashlib
import json
import math
import os
import time
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


def _metric_label(metric: str) -> str:
    """Build a pretty y-axis label for a metric, prefixed with 'Avg.' (LaTeX-safe)."""
    name_map = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "loss": "loss",
        "train_perplexity": "train perplexity",
    }
    base = name_map.get(metric, metric).replace("_", " ")
    if base == "accuracy":
        return r"Avg. accuracy (\%)"
    if base == "train perplexity":
        return r"Avg. training perplexity"
    return f"Avg. {base}"


def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


# ---------------------------
# On-disk cache: histories
# ---------------------------
def _cache_path_for_run(cache_dir: str, dataset: str, run_id: str) -> str:
    """Compute cache filepath for a run history under a dataset."""
    return os.path.join(cache_dir, dataset, f"{run_id}.pkl")


def _load_history_cached(run, cache_dir: str | None, dataset: str):
    """(Legacy) Load a run's history from cache if available; otherwise fetch and cache."""
    if not cache_dir:
        return run.history()
    path = _cache_path_for_run(cache_dir, dataset, run.id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass  # fall back to refetch if cache is corrupt
    df = run.history()
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


def _load_history_cached_by_id(
    api, project_path: str, run_id: str, cache_dir: str | None, dataset: str
):
    """History cache keyed by run_id; only hits the API on a cache miss."""
    if not cache_dir:
        return api.run(f"{project_path}/{run_id}").history()
    path = _cache_path_for_run(cache_dir, dataset, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    run = api.run(f"{project_path}/{run_id}")
    df = run.history()
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


# ---------------------------
# On-disk cache: run listings
# ---------------------------
def _runs_index_path(cache_dir: str, dataset: str, keyhash: str) -> str:
    return os.path.join(cache_dir, dataset, "_runs_index", f"{keyhash}.pkl")


def _filters_key(project_path: str, filters: dict) -> str:
    """Stable key for (project_path, filters) irrespective of dict ordering."""
    canonical = json.dumps(
        {"project": project_path, "filters": filters}, sort_keys=True, default=str
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _list_runs_cached(
    api,
    project_path: str,
    filters: dict,
    cache_dir: str | None,
    dataset: str,
    max_age_hours: int = 24,
):
    """
    Cache the listing of runs for (project_path, filters).
    Returns a list of {'id': ..., 'config': {...}} dicts.
    """
    if not cache_dir:
        return [
            {"id": r.id, "config": dict(getattr(r, "config", {}))}
            for r in api.runs(project_path, filters=filters)
        ]

    keyhash = _filters_key(project_path, filters)
    path = _runs_index_path(cache_dir, dataset, keyhash)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = time.time()

    if os.path.exists(path):
        try:
            payload = pd.read_pickle(path)  # {'_fetched_at': ts, 'runs': [...]}
            if isinstance(payload, dict) and "_fetched_at" in payload:
                if now - payload["_fetched_at"] <= max_age_hours * 3600:
                    return payload["runs"]
        except Exception:
            pass  # refresh below

    runs = api.runs(project_path, filters=filters)
    records = [{"id": r.id, "config": dict(getattr(r, "config", {}))} for r in runs]
    try:
        pd.to_pickle({"_fetched_at": now, "runs": records}, path)
    except Exception:
        pass
    return records


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
    combo_key: str = "num_stages",  # "num_stages" or "num_subdomains"
    cache_dir: str | None = None,  # on-disk cache root; per-dataset subdirs are created
    runs_cache_max_age_hours: int = 24,  # TTL for cached api.runs listings
):
    """
    Fetch runs with general + optional per-size SGD filters, then plot time-series grid.
    x_limits: optional dict mapping x_axis names to (min, max) tuples.
    y_limits: optional dict mapping metric names to (min, max) tuples.
    combo_key: which configuration key to use for combos/legend/grouping.
    cache_dir: if provided, caches per-run histories as pickles under {cache_dir}/{dataset}/<run_id>.pkl
               and caches run listings under {cache_dir}/{dataset}/_runs_index/<hash>.pkl
    runs_cache_max_age_hours: TTL for the run listing cache.
    """
    extra_filters = extra_filters or {}
    sgd_filters = sgd_filters or {}
    api = wandb.Api()

    dataset_name = filters_base.get("config.dataset_name", "dataset")

    # Collect runs (cache the listing)
    runs_all = []
    for size in batch_sizes:
        base_f = {**filters_base, f"config.{base_key}": size, **extra_filters}
        runs_gen = _list_runs_cached(
            api, project_path, base_f, cache_dir, dataset_name, runs_cache_max_age_hours
        )
        runs_gen = [
            m
            for m in runs_gen
            if m["config"].get(base_key) == size
            and m["config"].get("optimizer", "").lower() != "sgd"
        ]
        runs_all.extend(runs_gen)

        if size in sgd_filters:
            lr_f = sgd_filters[size]
            sgd_f = {
                **base_f,
                "config.optimizer": "sgd",
                # Keep SGD as the serial baseline regardless of combo_key
                "config.num_stages": 1,
                "config.num_subdomains": 1,
                **{f"config.{k}": v for k, v in lr_f.items()},
            }
            runs_sgd = _list_runs_cached(
                api,
                project_path,
                sgd_f,
                cache_dir,
                dataset_name,
                runs_cache_max_age_hours,
            )
            runs_sgd = [m for m in runs_sgd if m["config"].get(base_key) == size]
            runs_all.extend(runs_sgd)

    print(f"Collected total of {len(runs_all)} runs (from cache when possible).")

    @lru_cache(maxsize=None)
    def get_history(run_id: str):
        # Uses on-disk cache per dataset, then in-process LRU
        return _load_history_cached_by_id(
            api, project_path, run_id, cache_dir, dataset_name
        )

    # Build combos using the selected combo_key
    combos = sorted(
        {
            (m["config"].get("optimizer", "unk"), m["config"].get(combo_key, "unk"))
            for m in runs_all
        },
        key=lambda x: (str(x[0]), _safe_int(x[1])),
    )

    # Legend labels; keep "N=" where N refers to combo_key
    def _legend_label(opt, nval):
        n_str = "?" if nval in ("unk", None) else str(nval)
        return rf"${latex_opt(str(opt).upper())}\;\mid\;N={n_str}$"

    labels_full = [_legend_label(opt, nval) for opt, nval in combos]
    labels_full = [
        label.replace("APTS", "SAPTS") if "APTS" in label else label
        for label in labels_full
    ]

    colour_map = {combo: f"C{i}" for i, combo in enumerate(combos)}

    # Organize runs by (size, combo) using the selected combo_key
    runs_by = {}
    for m in runs_all:
        bs = m["config"].get(base_key)
        combo = (m["config"].get("optimizer", "unk"), m["config"].get(combo_key, "unk"))
        runs_by.setdefault((bs, combo), []).append(m)

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
                for m in runs_by.get((size, combo), []):
                    hist = get_history(m["id"])
                    if x_axis in hist.columns and metric in hist.columns:
                        df = hist[[x_axis, metric]].dropna()
                        if not df.empty:
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
                ax.set_ylabel(_metric_label(metric))
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

    # Choose which field defines combos/legend/grouping:
    combo_key = "num_subdomains"  # or: "num_stages"

    # On-disk cache root (per-dataset subdirectories are auto-created)
    cache_dir = os.path.join(base_dir, ".cache")

    datasets = ["tinyshakespeare"]
    configs = {
        "mnist": {
            "filters": {"config.model_name": "simple_cnn"},
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.01},
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
            "filters": {"config.model_name": "big_resnet"},
            "sgd_filters_strong": {
                256: {"learning_rate": 0.01},
                512: {"learning_rate": 0.1},
                1024: {"learning_rate": 0.1},
            },
            "sgd_filters_weak": {
                256: {"learning_rate": 0.01},
                512: {"learning_rate": 0.1},
                1024: {"learning_rate": 0.1},
            },
            "x_limits": {"grad_evals": (0, 40), "running_time": (0, 2000)},
            "y_limits": {"loss": (0, 5.0), "accuracy": (0, 100)},
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
                    combo_key=combo_key,
                    cache_dir=cache_dir,
                    runs_cache_max_age_hours=24,
                )


if __name__ == "__main__":
    main()
