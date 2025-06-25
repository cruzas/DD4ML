#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate per-dataset scaling tables (mean ± std, speedup, efficiency)
for both strong and weak regimes, grouping by optimizer, batch size (or eff batch size),
and number of subdomains, saving each as a .txt file.
"""

import os
import pprint
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced

BASELINE_N = 2


def prepare_scaling_table(gdf, group_cols, baseline_n=BASELINE_N):
    """
    Build a tidy DataFrame with speedup & efficiency based on baseline_n.
    group_cols: list of (key, abbr) pairs to rename and order.
    """
    # rename config and summary columns
    rename_map = {f"config_{k}": abbr for k, abbr in group_cols}
    rename_map.update(
        {
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
    df = gdf.rename(columns=rename_map)

    # identify abbr for batch and subdomains
    abbr_bs = next(abbr for k, abbr in group_cols if "batch" in k)
    abbr_N = next(abbr for k, abbr in group_cols if k == "num_subdomains")

    # baseline times at N=baseline_n per bs
    baseline = df[df[abbr_N] == baseline_n].set_index(abbr_bs)["time_mean"].to_dict()
    df["baseline_time"] = df[abbr_bs].map(baseline)

    # compute scaling metrics
    df["speedup"] = df["baseline_time"] / df["time_mean"]
    df["efficiency"] = df["speedup"] / (df[abbr_N] / baseline_n)

    # final order: grouped columns, metrics, scaling
    cols = [abbr for _, abbr in group_cols] + [
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


def process_scaling(
    proj, dataset, size_val, scale_type, base_key, bs_abbr, group_keys, group_abbrs
):
    """
    Common routine for either strong or weak scaling.
    scale_type: 'strong' or 'weak'
    base_key: config key for size filter, e.g. 'batch_size' or 'effective_batch_size'
    bs_abbr: corresponding abbreviation in filename
    group_keys: list of config keys for grouping
    group_abbrs: list of abbreviations for grouping columns
    """
    filters = {
        "config.dataset_name": dataset,
        f"config.{base_key}": size_val,
    }

    _, gdf = analyze_wandb_runs_advanced(
        project_path=proj,
        filters=filters,
        group_by=group_keys,
        group_by_abbr=group_abbrs,
        metrics=["loss", "accuracy", "grad_evals", "running_time"],
        aggregate="mean",
        mad_threshold=3.0,
    )
    if gdf is None:
        pprint.pprint(
            f"No {scale_type}-scaling runs for {dataset}, {bs_abbr}={size_val}"
        )
        return

    table = prepare_scaling_table(
        gdf,
        group_cols=list(zip(group_keys, group_abbrs)),
    )
    fname = f"{dataset}_{group_abbrs[1]}_{size_val}_{scale_type}_scaling.txt"
    save_path = os.path.join(
        os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures"), fname
    )
    with open(save_path, "w") as f:
        f.write(table.to_string(index=False, float_format="%.4f"))
    print(f"Saved {scale_type}-scaling table to {save_path}")


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
):
    proj = f"{entity}/{project}"
    datasets = ["mnist"]
    strong_batch_sizes = [1024, 2048, 4096]
    weak_batch_sizes = [128, 256, 512]  # can differ from strong sizes

    # strong scaling loop
    for dataset in datasets:
        for bs in strong_batch_sizes:
            process_scaling(
                proj,
                dataset,
                bs,
                scale_type="strong",
                base_key="batch_size",
                bs_abbr="bs",
                group_keys=["optimizer", "batch_size", "num_subdomains"],
                group_abbrs=["opt", "bs", "N"],
            )

    # weak scaling loop
    for dataset in datasets:
        for effbs in weak_batch_sizes:
            process_scaling(
                proj,
                dataset,
                effbs,
                scale_type="weak",
                base_key="effective_batch_size",
                bs_abbr="effbs",
                group_keys=["optimizer", "effective_batch_size", "num_subdomains"],
                group_abbrs=["opt", "effbs", "N"],
            )


if __name__ == "__main__":
    main()
