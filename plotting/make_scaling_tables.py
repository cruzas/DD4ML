#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate per-dataset, per-batch-size scaling summary tables (mean ± std, speedup, efficiency)
from wandb runs, saving each as a .txt file.
"""

import os
import pprint
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced


def prepare_scaling_table(gdf, batch_col, subdomains_col):
    """
    From grouped_df with summary_*_{mean,std,count} columns,
    compute speedup & efficiency (baseline at N=2) and return a tidy DataFrame.
    """
    df = gdf.rename(
        columns={
            batch_col: "bs",
            subdomains_col: "N",
            "summary_loss_mean": "loss_mean",
            "summary_loss_std": "loss_std",
            "summary_accuracy_mean": "acc_mean",
            "summary_accuracy_std": "acc_std",
            "summary_grad_evals_mean": "evals_mean",
            "summary_grad_evals_std": "evals_std",
            "summary_running_time_mean": "time_mean",
            "summary_running_time_std": "time_std",
        }
    )

    # baseline time per batch size = time at N=2
    BASELINE_N = 2
    baseline_times = df[df["N"] == BASELINE_N].set_index("bs")["time_mean"].to_dict()
    df["baseline_time"] = df["bs"].map(baseline_times)

    # compute speedup & efficiency
    # speedup = T(2)/T(N)
    df["speedup"] = df["baseline_time"] / df["time_mean"]
    # efficiency = speedup / (N/2)
    df["efficiency"] = df["speedup"] / (df["N"] / BASELINE_N)

    cols = [
        "bs",
        "N",
        "loss_mean",
        "loss_std",
        "acc_mean",
        "acc_std",
        "evals_mean",
        "evals_std",
        "time_mean",
        "time_std",
        "speedup",
        "efficiency",
    ]
    return df[cols]


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
):
    # where to save .txt tables
    base_save = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    os.makedirs(base_save, exist_ok=True)

    proj = f"{entity}/{project}"
    group_by = ["effective_batch_size", "num_subdomains"]
    group_by_abbr = ["bs", "N"]

    datasets = ["mnist"]
    batch_sizes = [128, 256, 512]

    metrics = ["loss", "accuracy", "grad_evals", "running_time"]
    aggregate = "mean"
    mad_threshold = 3.0

    for dataset, bs in product(datasets, batch_sizes):
        # build filters per (dataset, batch_size)
        filters = {
            "config.dataset_name": dataset,
            "config.effective_batch_size": bs,
        }

        # fetch & group
        df, gdf = analyze_wandb_runs_advanced(
            project_path=proj,
            filters=filters,
            group_by=group_by,
            group_by_abbr=group_by_abbr,
            metrics=metrics,
            aggregate=aggregate,
            mad_threshold=mad_threshold,
        )

        if gdf is None:
            pprint.pprint(f"No runs found for {dataset}, bs={bs}")
            continue

        # prepare the scaling table
        table = prepare_scaling_table(
            gdf,
            batch_col="config_effective_batch_size",
            subdomains_col="config_num_subdomains",
        )

        # save to .txt
        fname = f"{dataset}_bs_{bs}_scaling.txt"
        save_path = os.path.join(base_save, fname)
        with open(save_path, "w") as f:
            f.write(table.to_string(index=False, float_format="%.4f"))

        print(f"Saved scaling table to {save_path}")


if __name__ == "__main__":
    main()
