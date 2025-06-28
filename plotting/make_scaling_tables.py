#!/usr/bin/env python3
"""
make_scaling_tables.py

Generate two summary tables (mean ± std, sample count, speedup, efficiency) for strong and weak scaling,
across all specified batch sizes and datasets, grouping by optimizer and subdomains.
Each scaling regime writes a single .txt file including dataset name.
"""

import os
import pprint
from itertools import product

import pandas as pd

from plotting.analysis import analyze_wandb_runs_advanced


def prepare_scaling_table(gdf, group_cols):
    """
    Build a tidy DataFrame with dynamic speedup & efficiency.
    Include sample count for time metric and baseline at minimal N per subgroup.
    group_cols: list of (key, abbr) pairs to rename and order.
    """
    # rename config, summary and count columns
    rename_map = {f"config_{k}": abbr for k, abbr in group_cols}
    rename_map.update(
        {
            "summary_loss_mean": "loss_mean",
            "summary_loss_std": "loss_std",
            # "summary_accuracy_mean": "acc_mean",
            # "summary_accuracy_std": "acc_std",
            "summary_grad_evals_mean": "evals_mean",
            "summary_grad_evals_std": "evals_std",
            "summary_running_time_mean": "time_mean",
            "summary_running_time_std": "time_std",
            # include count of runs used for time
            "summary_running_time_count": "sample_count",
        }
    )
    df = gdf.rename(columns=rename_map)

    # identify abbr for N and grouping keys (excluding N)
    abbr_N = next(abbr for k, abbr in group_cols if k == "num_subdomains")
    abbr_group = [abbr for k, abbr in group_cols if k != "num_subdomains"]

    # compute minimal N per subgroup
    group_min = (
        df.groupby(abbr_group)[abbr_N]
        .min()
        .reset_index()
        .rename(columns={abbr_N: "min_N"})
    )

    # baseline maps
    baseline_time = {}
    baseline_n = {}
    for _, row in group_min.iterrows():
        key = tuple(row[abbr] for abbr in abbr_group)
        min_n = row["min_N"]
        mask = df[abbr_N] == min_n
        for abbr in abbr_group:
            mask &= df[abbr] == row[abbr]
        baseline_time[key] = df.loc[mask, "time_mean"].iloc[0]
        baseline_n[key] = min_n

    # map baseline values and N
    df["baseline_time"] = df.apply(
        lambda r: baseline_time[tuple(r[abbr] for abbr in abbr_group)], axis=1
    )
    df["baseline_N"] = df.apply(
        lambda r: baseline_n[tuple(r[abbr] for abbr in abbr_group)], axis=1
    )

    # compute scaling metrics
    df["speedup"] = df["baseline_time"] / df["time_mean"]
    df["efficiency"] = df["speedup"] / (df[abbr_N] / df["baseline_N"])

    # final column order
    cols = (
        abbr_group
        + [abbr_N, "sample_count"]
        + [
            "loss_mean",
            "loss_std",
            # "acc_mean",
            # "acc_std",
            "evals_mean",
            "evals_std",
            "time_mean",
            "time_std",
            "speedup",
            "efficiency",
        ]
    )
    return df[cols]


def collect_gdf_all(
    proj,
    datasets,
    sizes,
    base_key,
    group_keys,
    group_abbrs,
    # metrics=["loss", "accuracy", "grad_evals", "running_time"],
    metrics=["loss", "grad_evals", "running_time"],
    aggregate="mean",
    mad_threshold=1e99,
):
    """
    Run analyze_wandb_runs_advanced for each (dataset, size) combination,
    collect all grouped DataFrames and tag with dataset.
    """
    all_gdfs = []
    for dataset, size_val in product(datasets, sizes):
        filters = {
            "config.dataset_name": dataset,
            f"config.{base_key}": size_val,
            # "config.model_name": "minigpt",
            "config.model_name": "simple_resnet",
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
            # tag dataset for filename and table
            gdf["config_dataset_name"] = dataset
            all_gdfs.append(gdf)
        else:
            pprint.pprint(f"No runs: {base_key}={size_val}, dataset={dataset}")
    return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="thesis_results",
):
    proj = f"{entity}/{project}"

    mnist = False
    cifar10 = True
    tinyshakespeare = False

    assert not (mnist and cifar10 and tinyshakespeare), "Only one dataset can be True"

    if mnist:
        datasets = ["mnist"]
        strong_sizes = []
        weak_sizes = [128, 256, 512]
        strong_sizes = [1024, 2048, 4096]
    elif cifar10:
        datasets = ["cifar10"]
        weak_sizes = [512, 1024, 2048]
        strong_sizes = [4096, 8192, 16384]
    else:
        datasets = ["tinyshakespeare"]
        weak_sizes = [128, 256, 512]
        strong_sizes = [1024, 2048, 4096]

    out_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures")
    os.makedirs(out_dir, exist_ok=True)

    strong = True
    weak = True
    if strong:
        # strong scaling
        gdf_strong = collect_gdf_all(
            proj,
            datasets,
            strong_sizes,
            base_key="batch_size",
            group_keys=["optimizer", "batch_size", "num_subdomains"],
            group_abbrs=["opt", "bs", "N"],
        )
        if gdf_strong is not None:
            table_strong = prepare_scaling_table(
                gdf_strong,
                list(
                    zip(
                        ["optimizer", "batch_size", "num_subdomains"],
                        ["opt", "bs", "N"],
                    )
                ),
            )
            ds_name = "_".join(datasets)
            path = os.path.join(out_dir, f"{ds_name}_strong_scaling.txt")
            with open(path, "w") as f:
                f.write(table_strong.to_string(index=False, float_format="%.4f"))
            print(f"Saved strong scaling to {path}")
    if weak:
        # weak scaling
        gdf_weak = collect_gdf_all(
            proj,
            datasets,
            weak_sizes,
            base_key="effective_batch_size",
            group_keys=["optimizer", "effective_batch_size", "num_subdomains"],
            group_abbrs=["opt", "effbs", "N"],
        )
        if gdf_weak is not None:
            table_weak = prepare_scaling_table(
                gdf_weak,
                list(
                    zip(
                        ["optimizer", "effective_batch_size", "num_subdomains"],
                        ["opt", "ebs", "N"],
                    )
                ),
            )
            ds_name = "_".join(datasets)
            path = os.path.join(out_dir, f"{ds_name}_weak_scaling.txt")
            with open(path, "w") as f:
                f.write(table_weak.to_string(index=False, float_format="%.4f"))
            print(f"Saved weak scaling to {path}")


if __name__ == "__main__":
    main()
