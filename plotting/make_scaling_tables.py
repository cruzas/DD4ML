#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate summary tables (mean ± std, sample count, speedup, efficiency—and accuracy
for selected datasets) for strong and weak scaling across specified batch sizes
and datasets, grouping by optimiser and subdomains.
"""

import os
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced


def prepare_scaling_table(gdf, group_cols, include_acc=False):
    rename_map = {f"config_{k}": abbr for k, abbr in group_cols}
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

    abbr_N = next(abbr for k, abbr in group_cols if k == "num_subdomains")
    abbr_group = [abbr for k, abbr in group_cols if k != "num_subdomains"]

    mins = (
        df.groupby(abbr_group)[abbr_N]
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
        baseline[key] = {
            "time": df.loc[mask, "time_mean"].iloc[0],
            "N": row["min_N"],
        }

    df["baseline_time"] = df.apply(
        lambda r: baseline[tuple(r[a] for a in abbr_group)]["time"], axis=1
    )
    df["baseline_N"] = df.apply(
        lambda r: baseline[tuple(r[a] for a in abbr_group)]["N"], axis=1
    )

    df["speedup"] = df["baseline_time"] / df["time_mean"]
    df["efficiency"] = df["speedup"] / (df[abbr_N] / df["baseline_N"])

    # Loss: 3 decimals
    for c in ("loss_mean", "loss_std"):
        df[c] = df[c].map(lambda x: f"{x:.3f}")

    # Accuracy: 2 decimals
    if include_acc:
        for c in ("acc_mean", "acc_std"):
            df[c] = df[c].map(lambda x: f"{x:.2f}")

    # Grad evaluations: 2 decimals
    for c in ("evals_mean", "evals_std"):
        df[c] = df[c].map(lambda x: f"{x:.2f}")

    # Time: 2 decimals
    for c in ("time_mean", "time_std"):
        df[c] = df[c].map(lambda x: f"{x:.2f}")

    # Speedup: 2 decimals
    df["speedup"] = df["speedup"].map(lambda x: f"{x:.2f}")

    # Efficiency: percentage with 2 decimals
    df["efficiency"] = df["efficiency"].map(lambda x: f"{x*100:.2f}")

    cols = (
        abbr_group
        + [abbr_N, "sample_count", "loss_mean", "loss_std"]
        + (["acc_mean", "acc_std"] if include_acc else [])
        + ["evals_mean", "evals_std", "time_mean", "time_std", "speedup", "efficiency"]
    )
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
):
    """
    For each (dataset, size):
      1) fetch general runs (dropping any SGD),
      2) fetch SGD runs if sgd_filters[size] exists,
      3) concatenate both.
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

        # 1) general runs (exclude SGD)
        _, gdf_gen = analyze_wandb_runs_advanced(
            project_path=proj,
            filters=base,
            group_by=group_keys,
            group_by_abbr=group_abbrs,
            metrics=list(metrics),
            aggregate=aggregate,
            mad_threshold=mad_threshold,
        )
        if gdf_gen is not None:
            gdf_gen = gdf_gen[gdf_gen["config_optimizer"] != "sgd"]

        # 2) SGD runs if user provided a lr for this size
        gdf_sgd = None
        if size_val in sgd_filters:
            lr_filter = sgd_filters[size_val]
            base_sgd = {**base, **lr_filter, "config.optimizer": "sgd"}
            _, gdf_sgd = analyze_wandb_runs_advanced(
                project_path=proj,
                filters=base_sgd,
                group_by=group_keys,
                group_by_abbr=group_abbrs,
                metrics=list(metrics),
                aggregate=aggregate,
                mad_threshold=mad_threshold,
            )

        # 3) combine
        pieces = [df for df in (gdf_gen, gdf_sgd) if df is not None]
        if not pieces:
            print(f"No runs: {base_key}={size_val}, dataset={dataset}")
            continue

        gdf = pd.concat(pieces, ignore_index=True)
        gdf["config_dataset_name"] = dataset
        all_gdfs.append(gdf)

    return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
):
    proj = f"{entity}/{project}"
    out_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    os.makedirs(out_dir, exist_ok=True)

    keep_track_of_acc = ["mnist", "cifar10"]

    # select dataset
    choice = "mnist"  # "mnist", "cifar10", "tinyshakespeare", "poisson2d"
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
            "strong_sizes": [2048, 4096, 8192],
            "weak_sizes": [256, 512, 1024],
            "filters": {"config.model_name": "simple_resnet"},
            "sgd_filters_strong": {
                2048: {"config.learning_rate": 0.1},
                4096: {"config.learning_rate": 0.1},
                8192: {"config.learning_rate": 0.1},
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

    regimes = {
        "strong": {
            "sizes": cfg["strong_sizes"],
            "base_key": "batch_size",
            "group_keys": ["optimizer", "batch_size", "num_subdomains"],
            "group_abbrs": ["opt", "bs", "N"],
            "extra_filters": cfg["filters"],
            "sgd_filters": cfg["sgd_filters_strong"],
        },
        "weak": {
            "sizes": cfg["weak_sizes"],
            "base_key": "effective_batch_size",
            "group_keys": ["optimizer", "effective_batch_size", "num_subdomains"],
            "group_abbrs": ["opt", "ebs", "N"],
            "extra_filters": cfg["filters"],
            "sgd_filters": cfg["sgd_filters_weak"],
        },
    }

    for name, params in regimes.items():
        gdf = collect_gdf_all(
            proj,
            cfg["datasets"],
            params["sizes"],
            params["base_key"],
            params["group_keys"],
            params["group_abbrs"],
            extra_filters=params["extra_filters"],
            sgd_filters=params["sgd_filters"],
            metrics=base_metrics,
        )
        if gdf is None:
            continue

        table = prepare_scaling_table(
            gdf,
            list(zip(params["group_keys"], params["group_abbrs"])),
            include_acc=include_acc,
        )
        tag = "_".join(cfg["datasets"])
        fname = f"{tag}_{name}_scaling.txt"
        path = os.path.join(out_dir, fname)

        with open(path, "w") as f:
            f.write(table.to_string(index=False))
        print(f"Saved {name} scaling to {path}")


if __name__ == "__main__":
    main()
