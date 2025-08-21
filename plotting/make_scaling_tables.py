#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate summary tables (mean ± std, sample count, speedup, efficiency—and accuracy
for selected datasets) for strong and weak scaling across specified batch sizes
and datasets, grouping by optimiser and a parallelism key.

New:
- Reuses on-disk caches produced by time_series_grid.py when available:
  * run listings: {cache_dir}/{dataset}/_runs_index/<hash>.pkl
  * histories:    {cache_dir}/{dataset}/{run_id}.pkl
- Falls back to analyze_wandb_runs_advanced when caches are missing.

Behaviour:
- Non-SGD runs respect --parallel-key (num_subdomains OR num_stages).
- SGD runs are ALWAYS grouped and printed with N = num_subdomains.
"""

import argparse
import hashlib
import json
import os
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced

PARALLEL_KEYS = {"num_subdomains", "num_stages"}


# ---------------------------
# Cache helpers (mirror time_series_grid.py)
# ---------------------------
def _filters_key(project_path: str, filters: dict) -> str:
    canonical = json.dumps(
        {"project": project_path, "filters": filters}, sort_keys=True, default=str
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _runs_index_path(cache_dir: str, dataset: str, keyhash: str) -> str:
    return os.path.join(cache_dir, dataset, "_runs_index", f"{keyhash}.pkl")


def _history_path(cache_dir: str, dataset: str, run_id: str) -> str:
    return os.path.join(cache_dir, dataset, f"{run_id}.pkl")


def _try_load_cached_listing(
    project_path: str, filters: dict, cache_dir: Optional[str], dataset: str
):
    if not cache_dir:
        return None
    keyhash = _filters_key(project_path, filters)
    path = _runs_index_path(cache_dir, dataset, keyhash)
    if not os.path.exists(path):
        return None
    try:
        payload = pd.read_pickle(
            path
        )  # {'_fetched_at': ts, 'runs': [{'id':..,'config':{...}}, ...]}
        if isinstance(payload, dict) and "runs" in payload:
            return payload["runs"]
    except Exception:
        return None
    return None


def _try_load_cached_history(
    cache_dir: Optional[str], dataset: str, run_id: str
) -> Optional[pd.DataFrame]:
    if not cache_dir:
        return None
    path = _history_path(cache_dir, dataset, run_id)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _last_non_nan(series: pd.Series):
    if series is None or series.empty:
        return np.nan
    # find last valid entry
    idx = series.last_valid_index()
    if idx is None:
        return np.nan
    return series.loc[idx]


def _build_gdf_via_cache(
    project_path: str,
    dataset: str,
    filters: Dict,
    group_keys: List[str],  # e.g., ["optimizer", "batch_size", "num_stages"]
    metrics: List[str],  # e.g., ["loss","grad_evals","running_time","accuracy"?]
    cache_dir: Optional[str],
    exclude_sgd: bool = False,  # for non-SGD path
) -> Optional[pd.DataFrame]:
    """
    Attempt to build the grouped summary DataFrame using ONLY on-disk caches.
    Returns None if any critical cache entries are missing (so caller may fall back).
    Produces columns like analyze_wandb_runs_advanced's gdf: 'config_<k>' and 'summary_<metric>_mean/std'.
    """
    # 1) Load cached listing for these filters
    listing = _try_load_cached_listing(project_path, filters, cache_dir, dataset)
    if listing is None or len(listing) == 0:
        return None

    # 2) Collect per-run scalars using cached histories
    run_rows = []
    for rec in listing:
        cfg = rec.get("config", {}) or {}
        opt = (cfg.get("optimizer") or cfg.get("Optimizer") or "").lower()
        if exclude_sgd and opt == "sgd":
            continue

        hist = _try_load_cached_history(cache_dir, dataset, rec["id"])
        if hist is None or hist.empty:
            # If any history is missing, we choose to SKIP that run rather than hit the API.
            # If skipping yields no data, we fall back to the API at the caller level.
            continue

        # Prepare one row per run with run-level scalars (last non-NaN per metric)
        row = {}
        # Config columns expected downstream
        for k in set(group_keys) | {
            "dataset_name",
            "num_stages",
            "num_subdomains",
            "batch_size",
            "effective_batch_size",
            "optimizer",
        }:
            if k in cfg:
                row[f"config_{k}"] = cfg[k]
        # Metrics: take last observed value in the time series
        for m in metrics:
            if m in hist.columns:
                row[f"run_{m}"] = float(_last_non_nan(hist[m]))
            else:
                row[f"run_{m}"] = np.nan

        run_rows.append(row)

    if not run_rows:
        return None

    runs_df = pd.DataFrame(run_rows)

    # Ensure required config columns exist even if absent in some runs
    for k in group_keys:
        col = f"config_{k}"
        if col not in runs_df:
            runs_df[col] = np.nan

    # 3) Group by requested config keys; compute mean/std/count across runs
    g = runs_df.groupby([f"config_{k}" for k in group_keys], dropna=False)

    out = []
    for grp_vals, sub in g:
        rec = {
            f"config_{k}": v
            for k, v in zip(
                group_keys, (grp_vals if isinstance(grp_vals, tuple) else (grp_vals,))
            )
        }
        count = len(sub)
        for m in metrics:
            vals = sub[f"run_{m}"].astype(float)
            rec[f"summary_{m}_mean"] = float(np.nanmean(vals)) if count else np.nan
            rec[f"summary_{m}_std"] = (
                float(np.nanstd(vals, ddof=0)) if count else np.nan
            )
        # a count column aligned with prepare_scaling_table expectations (uses running_time_count)
        rec["summary_running_time_count"] = int(count)
        out.append(rec)

    gdf = pd.DataFrame(out)

    # Also carry dataset name if present in filters/config
    if "config.dataset_name" in filters:
        gdf["config_dataset_name"] = filters["config.dataset_name"]

    return gdf if not gdf.empty else None


def prepare_scaling_table(gdf, group_cols, include_acc=False):
    """
    Parameters
    ----------
    gdf : pd.DataFrame
        Output of collect_gdf_all (already contains 'config_N' = unified parallel degree).
    group_cols : list[tuple[str, str]]
        (original_key, abbr) pairs for columns to show, where the parallel column MUST be ("N","N").
        Example: [("optimizer","opt"), ("batch_size","bs"), ("N","N")]
    include_acc : bool
        Whether to include accuracy columns.

    Returns
    -------
    pd.DataFrame : formatted table ready for writing.
    """
    # Rename config_* columns to their abbreviations
    rename_map = {f"config_{k}": abbr for k, abbr in group_cols}
    # Ensure the unified N is renamed (we create config_N in collect_gdf_all)
    rename_map["config_N"] = "N"

    rename_map.update(
        {
            "summary_loss_mean": "loss_mean",
            "summary_loss_std": "loss_std",
            "summary_grad_evals_mean": "evals_mean",
            "summary_grad_evals_std": "evals_std",
            "summary_running_time_mean": "time_mean",
            "summary_running_time_std": "time_std",
            "summary_running_time_count": "sample_count",
        }
    )
    if include_acc:
        rename_map.update(
            {
                "summary_accuracy_mean": "acc_mean",
                "summary_accuracy_std": "acc_std",
            }
        )

    df = gdf.rename(columns=rename_map)

    # Use the unified N everywhere
    abbr_N = "N"
    abbr_group = [abbr for _, abbr in group_cols if abbr != abbr_N]

    # Baseline: per (opt, bs/ebs) group, pick the minimum N
    mins = (
        df.groupby(abbr_group, dropna=False)[abbr_N]
        .min()
        .reset_index()
        .rename(columns={abbr_N: "min_N"})
    )

    baseline = {}
    for _, row in mins.iterrows():
        key = tuple(row[a] for a in abbr_group)
        mask = df[abbr_N].eq(row["min_N"])
        for a in abbr_group:
            mask &= df[a].eq(row[a])
        if not mask.any():
            continue
        baseline[key] = {
            "time": df.loc[mask, "time_mean"].iloc[0],
            "N": row["min_N"],
        }

    def _btime(r):
        return baseline.get(tuple(r[a] for a in abbr_group), {}).get(
            "time", float("nan")
        )

    def _bN(r):
        return baseline.get(tuple(r[a] for a in abbr_group), {}).get("N", float("nan"))

    df["baseline_time"] = df.apply(_btime, axis=1)
    df["baseline_N"] = df.apply(_bN, axis=1)

    # Speedup and efficiency
    df["speedup"] = df["baseline_time"] / df["time_mean"]
    df["efficiency"] = df["speedup"] / (df[abbr_N] / df["baseline_N"])

    # Formatting
    if "loss_mean" in df:
        df["loss_mean"] = df["loss_mean"].map(lambda x: f"{float(x):.3f}")
    if "loss_std" in df:
        df["loss_std"] = df["loss_std"].map(lambda x: f"{float(x):.3f}")

    if include_acc:
        if "acc_mean" in df:
            df["acc_mean"] = df["acc_mean"].map(lambda x: f"{float(x):.2f}")
        if "acc_std" in df:
            df["acc_std"] = df["acc_std"].map(lambda x: f"{float(x):.2f}")

    for c in ("evals_mean", "evals_std", "time_mean", "time_std"):
        if c in df:
            df[c] = df[c].map(lambda x: f"{float(x):.2f}")

    if "speedup" in df:
        df["speedup"] = df["speedup"].map(lambda x: f"{float(x):.2f}")
    if "efficiency" in df:
        # Express as percentage (no % symbol to keep LaTeX-safe if needed downstream)
        df["efficiency"] = df["efficiency"].map(lambda x: f"{100.0 * float(x):.2f}")

    cols = (
        abbr_group
        + [abbr_N, "sample_count", "loss_mean", "loss_std"]
        + (["acc_mean", "acc_std"] if include_acc else [])
        + ["evals_mean", "evals_std", "time_mean", "time_std", "speedup", "efficiency"]
    )
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def collect_gdf_all(
    proj,
    datasets,
    sizes,
    base_key,
    group_keys,
    group_abbrs,
    extra_filters=None,
    sgd_filters=None,
    metrics=("loss", "grad_evals", "running_time"),
    aggregate="mean",
    mad_threshold=1e99,
    parallel_key="num_subdomains",
    cache_dir: Optional[str] = None,  # NEW: path to time_series_grid cache root
):
    """
    For each (dataset, size):
      1) fetch general runs (non-SGD) grouped by user-selected parallel_key,
      2) fetch SGD runs (if provided) grouped by num_subdomains,
      3) concatenate both, and create a unified 'config_N' for printing/baselines.

    sgd_filters should be a dict mapping size_val -> {filter_key: filter_val, ...}.
    If cache_dir is provided, we first attempt to build gdfs from the on-disk caches;
    on failure, we fall back to analyze_wandb_runs_advanced.
    """
    extra_filters = extra_filters or {}
    sgd_filters = sgd_filters or {}
    all_gdfs = []

    for dataset, size_val in product(datasets, sizes):
        base = {
            "config.dataset_name": dataset,
            f"config.{base_key}": size_val,
        }
        base.update(extra_filters)

        # ---------- Non-SGD path ----------
        # Try cache first
        gdf_gen = _build_gdf_via_cache(
            project_path=proj,
            dataset=dataset,
            filters=base,
            group_keys=group_keys,  # includes the chosen parallel_key
            metrics=list(metrics) + (["accuracy"] if "accuracy" in metrics else []),
            cache_dir=cache_dir,
            exclude_sgd=True,
        )

        # Fallback to API-based analysis if cache is missing/insufficient
        if gdf_gen is None:
            _, gdf_gen = analyze_wandb_runs_advanced(
                project_path=proj,
                filters=base,
                group_by=group_keys,
                group_by_abbr=group_abbrs,
                metrics=list(metrics),
                aggregate=aggregate,
                mad_threshold=mad_threshold,
            )

        if gdf_gen is not None and "config_optimizer" in gdf_gen:
            gdf_gen = gdf_gen[gdf_gen["config_optimizer"] != "sgd"]

        # Attach unified N for the chosen parallel key
        if gdf_gen is not None and not gdf_gen.empty:
            if f"config_{parallel_key}" in gdf_gen:
                gdf_gen["config_N"] = gdf_gen[f"config_{parallel_key}"]
            elif "config_num_stages" in gdf_gen:
                gdf_gen["config_N"] = gdf_gen["config_num_stages"]
            elif "config_num_subdomains" in gdf_gen:
                gdf_gen["config_N"] = gdf_gen["config_num_subdomains"]

        # ---------- SGD path (force N = num_subdomains) ----------
        gdf_sgd = None
        if size_val in sgd_filters:
            lr_filter = sgd_filters[size_val]
            # The original code expects keys like "config.learning_rate" in sgd_filters
            base_sgd = {**base, **lr_filter, "config.optimizer": "sgd"}

            # group by with forced key
            sgd_group_keys = [k for k in group_keys if k != parallel_key] + [
                "num_subdomains"
            ]

            # Try cache first
            gdf_sgd = _build_gdf_via_cache(
                project_path=proj,
                dataset=dataset,
                filters=base_sgd,
                group_keys=sgd_group_keys,
                metrics=list(metrics) + (["accuracy"] if "accuracy" in metrics else []),
                cache_dir=cache_dir,
                exclude_sgd=False,  # we're specifically asking for SGD via filters
            )

            # Fallback to API-based analysis if needed
            if gdf_sgd is None:
                _, gdf_sgd = analyze_wandb_runs_advanced(
                    project_path=proj,
                    filters=base_sgd,
                    group_by=sgd_group_keys,
                    group_by_abbr=[
                        a for k, a in zip(group_keys, group_abbrs) if k != parallel_key
                    ]
                    + ["N"],
                    metrics=list(metrics),
                    aggregate=aggregate,
                    mad_threshold=mad_threshold,
                )

            if gdf_sgd is not None and not gdf_sgd.empty:
                if "config_num_subdomains" in gdf_sgd:
                    gdf_sgd["config_N"] = gdf_sgd["config_num_subdomains"]

        # ---------- Combine ----------
        pieces = [df for df in (gdf_gen, gdf_sgd) if df is not None and not df.empty]
        if not pieces:
            print(f"No runs: {base_key}={size_val}, dataset={dataset}")
            continue

        gdf = pd.concat(pieces, ignore_index=True)
        gdf["config_dataset_name"] = dataset
        all_gdfs.append(gdf)

    return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None


def build_argparser():
    p = argparse.ArgumentParser(description="Generate scaling tables.")
    p.add_argument(
        "--parallel-key",
        default="num_subdomains",  # default to stages (common for APTS/IP)
        choices=["num_subdomains", "num_stages"],
        help="Parallelism key for non-SGD runs. SGD always uses num_subdomains.",
    )
    p.add_argument(
        "--choice",
        default="poisson2d",
        choices=["mnist", "cifar10", "tinyshakespeare", "poisson2d"],
        help="Dataset configuration preset.",
    )
    p.add_argument(
        "--entity",
        default="cruzas-universit-della-svizzera-italiana",
        help="Weights & Biases entity.",
    )
    p.add_argument(
        "--project",
        default="thesis_results",
        help="Weights & Biases project.",
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.expanduser(
            "~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures/.cache"
        ),
        help="Path to time_series_grid.py cache root (if present).",
    )
    return p


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
    choice="poisson2d",
    parallel_key="num_subdomains",  # or num_stages
    cache_dir: Optional[str] = None,
):
    if parallel_key not in PARALLEL_KEYS:
        raise ValueError(f"parallel_key must be one of {PARALLEL_KEYS}")

    proj = f"{entity}/{project}"
    out_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    os.makedirs(out_dir, exist_ok=True)

    keep_track_of_acc = ["mnist", "cifar10"]

    configs = {
        "mnist": {
            "datasets": ["mnist"],
            "strong_sizes": [1024, 2048, 4096],
            "weak_sizes": [128, 256, 512],
            "filters": {"config.model_name": "simple_cnn"},
            "sgd_filters_strong": {
                1024: {"config.learning_rate": 0.10},
                2048: {"config.learning_rate": 0.10},
                4096: {"config.learning_rate": 0.10},
            },
            "sgd_filters_weak": {
                128: {"config.learning_rate": 0.01},
                256: {"config.learning_rate": 0.10},
                512: {"config.learning_rate": 0.10},
            },
        },
        "cifar10": {
            "datasets": ["cifar10"],
            "strong_sizes": [256, 512, 1024],
            "weak_sizes": [256, 512, 1024],
            "filters": {"config.model_name": "big_resnet"},
            "sgd_filters_strong": {
                256: {"config.learning_rate": 0.01},
                512: {"config.learning_rate": 0.1},
                1024: {"config.learning_rate": 0.1},
            },
            "sgd_filters_weak": {
                256: {"config.learning_rate": 0.01},
                512: {"config.learning_rate": 0.1},
                1024: {"config.learning_rate": 0.1},
            },
        },
        "tinyshakespeare": {
            "datasets": ["tinyshakespeare"],
            "strong_sizes": [1024, 2048, 4096],
            "weak_sizes": [128, 256, 512],
            "filters": {"config.model_name": "minigpt"},
            "sgd_filters_strong": {
                1024: {"config.learning_rate": 0.01},
                2048: {"config.learning_rate": 0.01},
                4096: {"config.learning_rate": 0.01},
            },
            "sgd_filters_weak": {
                128: {"config.learning_rate": 0.10},
                256: {"config.learning_rate": 0.10},
                512: {"config.learning_rate": 0.10},
            },
        },
        "poisson2d": {
            "datasets": ["poisson2d"],
            "strong_sizes": [128, 256, 512],
            "weak_sizes": [64, 128, 256],
            "filters": {"config.model_name": "pinn_ffnn"},
            "sgd_filters_strong": {
                128: {"config.learning_rate": 0.001},
                256: {"config.learning_rate": 0.10},
                512: {"config.learning_rate": 0.001},
            },
            "sgd_filters_weak": {
                64: {"config.learning_rate": 0.01},
                128: {"config.learning_rate": 0.001},
                256: {"config.learning_rate": 0.10},
            },
        },
    }

    cfg = configs[choice]
    include_acc = choice in keep_track_of_acc

    base_metrics = ["loss", "grad_evals", "running_time"]
    if include_acc:
        base_metrics.append("accuracy")

    base_key_abbr = {
        "strong": ("batch_size", "bs"),
        "weak": ("effective_batch_size", "ebs"),
    }

    regimes = {
        "strong": {
            "sizes": cfg["strong_sizes"],
            "extra_filters": cfg["filters"],
            "sgd_filters": cfg["sgd_filters_strong"],
        },
        "weak": {
            "sizes": cfg.get("weak_sizes", []),
            "extra_filters": cfg["filters"],
            "sgd_filters": cfg.get("sgd_filters_weak", {}),
        },
    }

    for name, params in regimes.items():
        if not params["sizes"]:
            continue

        base_key, base_abbr = base_key_abbr[name]

        # For retrieval/grouping:
        group_keys = ["optimizer", base_key, parallel_key]  # non-SGD path uses this
        group_abbrs = ["opt", base_abbr, "N"]

        gdf = collect_gdf_all(
            proj,
            cfg["datasets"],
            params["sizes"],
            base_key,
            group_keys,
            group_abbrs,
            extra_filters=params["extra_filters"],
            sgd_filters=params["sgd_filters"],
            metrics=base_metrics,
            parallel_key=parallel_key,  # pass through
            cache_dir=cache_dir,  # <— use cache if present
        )
        if gdf is None or gdf.empty:
            continue

        # For printing/baselines: use unified N (we expose ("N","N"))
        group_cols_for_print = [("optimizer", "opt"), (base_key, base_abbr), ("N", "N")]

        table = prepare_scaling_table(
            gdf,
            group_cols_for_print,
            include_acc=include_acc,
        )
        tag = "_".join(cfg["datasets"])
        par_tag = "ns" if parallel_key == "num_stages" else "nd"
        fname = f"{tag}_{name}_scaling_{par_tag}.txt"
        path = os.path.join(out_dir, fname)

        with open(path, "w") as f:
            f.write(table.to_string(index=False))
        print(f"Saved {name} scaling to {path}")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(
        entity=args.entity,
        project=args.project,
        choice=args.choice,
        parallel_key=args.parallel_key,
        cache_dir=args.cache_dir,
    )
