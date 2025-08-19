#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate summary tables (mean ± std, sample count, speedup, efficiency—and accuracy
for selected datasets) for strong and weak scaling across specified batch sizes
and datasets, grouping by optimiser and a parallelism key.

Behaviour:
- Non-SGD runs respect --parallel-key (num_subdomains OR num_stages).
- SGD runs are ALWAYS grouped and printed with N = num_subdomains.
"""

import argparse
import os
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced

PARALLEL_KEYS = {"num_subdomains", "num_stages"}


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
        df["loss_mean"] = df["loss_mean"].map(lambda x: f"{x:.3f}")
    if "loss_std" in df:
        df["loss_std"] = df["loss_std"].map(lambda x: f"{x:.3f}")

    if include_acc:
        if "acc_mean" in df:
            df["acc_mean"] = df["acc_mean"].map(lambda x: f"{x:.2f}")
        if "acc_std" in df:
            df["acc_std"] = df["acc_std"].map(lambda x: f"{x:.2f}")

    for c in ("evals_mean", "evals_std", "time_mean", "time_std"):
        if c in df:
            df[c] = df[c].map(lambda x: f"{x:.2f}")

    if "speedup" in df:
        df["speedup"] = df["speedup"].map(lambda x: f"{x:.2f}")
    if "efficiency" in df:
        df["efficiency"] = df["efficiency"].map(lambda x: f"{x*100:.2f}")

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
    parallel_key="num_stages",
):
    """
    For each (dataset, size):
      1) fetch general runs (non-SGD) grouped by user-selected parallel_key,
      2) fetch SGD runs (if provided) grouped by num_subdomains,
      3) concatenate both, and create a unified 'config_N' for printing/baselines.

    sgd_filters should be a dict mapping size_val -> {filter_key: filter_val, ...}.
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

        # 1) Non-SGD runs
        _, gdf_gen = analyze_wandb_runs_advanced(
            project_path=proj,
            filters=base,
            group_by=group_keys,  # includes user-selected parallel_key
            group_by_abbr=group_abbrs,  # ...and "N" as the abbr for that key
            metrics=list(metrics),
            aggregate=aggregate,
            mad_threshold=mad_threshold,
        )
        if gdf_gen is not None and "config_optimizer" in gdf_gen:
            gdf_gen = gdf_gen[gdf_gen["config_optimizer"] != "sgd"]
            # Set unified N from the chosen parallel key; fall back if absent
            if gdf_gen is not None and not gdf_gen.empty:
                if f"config_{parallel_key}" in gdf_gen:
                    gdf_gen["config_N"] = gdf_gen[f"config_{parallel_key}"]
                elif "config_num_stages" in gdf_gen:
                    gdf_gen["config_N"] = gdf_gen["config_num_stages"]
                elif "config_num_subdomains" in gdf_gen:
                    gdf_gen["config_N"] = gdf_gen["config_num_subdomains"]

        # 2) SGD runs (force grouping by num_subdomains)
        gdf_sgd = None
        if size_val in sgd_filters:
            lr_filter = sgd_filters[size_val]
            base_sgd = {**base, **lr_filter, "config.optimizer": "sgd"}

            # Replace the parallel key with num_subdomains for grouping
            sgd_group_keys = [k for k in group_keys if k != parallel_key] + [
                "num_subdomains"
            ]
            sgd_group_abbrs = [
                a for k, a in zip(group_keys, group_abbrs) if k != parallel_key
            ] + ["N"]

            _, gdf_sgd = analyze_wandb_runs_advanced(
                project_path=proj,
                filters=base_sgd,
                group_by=sgd_group_keys,
                group_by_abbr=sgd_group_abbrs,
                metrics=list(metrics),
                aggregate=aggregate,
                mad_threshold=mad_threshold,
            )
            if gdf_sgd is not None and not gdf_sgd.empty:
                # Unified N is always num_subdomains for SGD
                if "config_num_subdomains" in gdf_sgd:
                    gdf_sgd["config_N"] = gdf_sgd["config_num_subdomains"]

        # 3) Combine
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
        default="num_stages",  # default to stages (common for APTS/IP)
        choices=["num_subdomains", "num_stages"],
        help="Parallelism key for non-SGD runs. SGD always uses num_subdomains.",
    )
    p.add_argument(
        "--choice",
        default="cifar10",
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
    return p


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
    choice="cifar10",
    parallel_key="num_stages",
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
    )
