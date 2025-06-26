#!/usr/bin/env python3
"""
make_sgd_hyperparam_tables.py

Generate summary tables (mean ± std, sample count) for SGD hyperparameter tuning
across specified learning rates and batch sizes, grouping by dataset.
Each table writes a .txt file per dataset.
"""

import os
import pprint
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced


def prepare_hyperparam_table(gdf, group_cols):
    """
    Build a tidy DataFrame with summary statistics.
    group_cols: list of (key, abbr) pairs to rename and order.
    """
    rename_map = {f"config_{k}": abbr for k, abbr in group_cols}
    rename_map.update(
        {
            "summary_loss_mean": "loss_mean",
            "summary_loss_std": "loss_std",
            "summary_accuracy_mean": "acc_mean",
            "summary_accuracy_std": "acc_std",
            "summary_running_time_mean": "time_mean",
            "summary_running_time_std": "time_std",
            "summary_running_time_count": "sample_count",
        }
    )
    df = gdf.rename(columns=rename_map)

    cols = [abbr for _, abbr in group_cols] + [
        "sample_count",
        "loss_mean",
        "loss_std",
        "acc_mean",
        "acc_std",
        "time_mean",
        "time_std",
    ]
    return df[cols]


def collect_gdf_all(
    proj,
    dataset,
    lrs,
    bss,
    group_keys,
    group_abbrs,
    metrics=None,
    aggregate="mean",
    mad_threshold=3.0,
):
    """
    Run analyze_wandb_runs_advanced for each hyperparameter combination,
    collect all grouped DataFrames and tag with dataset.
    """
    metrics = metrics or ["loss", "accuracy", "running_time"]
    all_gdfs = []
    for lr, bs in product(lrs, bss):
        filters = {
            "config.dataset_name": dataset,
            "config.learning_rate": lr,
            "config.batch_size": bs,
        }
        _, gdf = analyze_wandb_runs_advanced(
            project_path=proj,
            filters=filters,
            group_by=group_keys,
            group_by_abbr=group_abbrs,
            metrics=metrics,
            aggregate=aggregate,
            mad_threshold=mad_threshold,
        )
        if gdf is not None:
            gdf["config_dataset_name"] = dataset
            all_gdfs.append(gdf)
        else:
            pprint.pprint(f"No runs: lr={lr}, bs={bs}, dataset={dataset}")
    return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None


def main(
    entity="cruzas-universit-della-svizzera-italiana", project="sgd_hyperparam_tuning"
):
    proj = f"{entity}/{project}"
    datasets = ["mnist"]
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_sizes = [128, 256, 512]

    out_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    os.makedirs(out_dir, exist_ok=True)

    for ds in datasets:
        gdf = collect_gdf_all(
            proj,
            ds,
            learning_rates,
            batch_sizes,
            group_keys=["learning_rate", "batch_size"],
            group_abbrs=["lr", "bs"],
        )
        if gdf is None:
            continue

        # make group_cols reusable by converting zip to list
        group_cols = list(zip(["learning_rate", "batch_size"], ["lr", "bs"]))
        table = prepare_hyperparam_table(gdf, group_cols)
        # for each bs: find lr that minimises loss_mean and that maximises acc_mean
        best = (
            table.groupby("bs")
            .apply(
                lambda df: {
                    "lr_min_loss": df.loc[df.loss_mean.idxmin(), "lr"],
                    "min_loss": df.loss_mean.min(),
                    "lr_max_acc": df.loc[df.acc_mean.idxmax(), "lr"],
                    "max_acc": df.acc_mean.max(),
                }
            )
            .apply(pd.Series)
            .reset_index()
        )

        print(best.to_string(index=False))

        fname = f"{ds}_sgd_tuning_summary.txt"
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            f.write(table.to_string(index=False, float_format="%.4f"))
        print(f"Saved hyperparameter table to {path}")


if __name__ == "__main__":
    main()
