import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

import wandb


def _abbreviate_val(val):
    if isinstance(val, bool):
        return "T" if val else "F"
    return val


def plot_averaged_time_series(
    project_path,
    filters=None,
    group_by=None,
    group_by_abbr=None,
    metric_name="loss",
    show_variance=True,
):
    """Plot time series data averaged across runs with same configuration"""

    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(group_by_abbr, str):
        group_by_abbr = [group_by_abbr]

    abbr_map = (
        dict(zip(group_by, group_by_abbr))
        if group_by and group_by_abbr and len(group_by_abbr) == len(group_by)
        else {}
    )

    grouped = {}
    for run in runs:
        if group_by:
            key = tuple(run.config.get(k, "unknown") for k in group_by)
            parts = []
            for k, v in zip(group_by, key):
                v2 = _abbreviate_val(v)
                parts.append(f"{abbr_map.get(k, k)}={v2}")
            label = " | ".join(parts)
        else:
            key, label = "all_runs", "All Runs"
        grouped.setdefault(key, {"runs": [], "label": label})["runs"].append(run)

    plt.figure(figsize=(12, 8))
    for info in grouped.values():
        histories = [
            h[["_step", metric_name]].dropna()
            for run in info["runs"]
            for h in [run.history()]
            if metric_name in h.columns
        ]
        if not histories:
            continue

        min_s = min(h["_step"].min() for h in histories)
        max_s = max(h["_step"].max() for h in histories)
        grid = np.linspace(min_s, max_s, 100)
        data = np.stack(
            [np.interp(grid, h["_step"], h[metric_name]) for h in histories]
        )
        m, s = data.mean(0), data.std(0)

        plt.plot(grid, m, label=f"{info['label']} (n={len(histories)})", linewidth=2)
        if show_variance and len(histories) > 1:
            plt.fill_between(grid, m - s, m + s, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Time (Averaged by Configuration)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_wandb_runs_advanced(
    project_path,
    filters=None,
    group_by=None,
    group_by_abbr=None,
    metrics=None,
    plot_type="scatter",
    show_variance=True,
    aggregate="mean",
    mad_threshold=3,
):
    """
    Analyze wandb runs with multi-key grouping, robust filtering, and variance analysis.

    Added:
        group_by_abbr: List of abbreviations for each key in `group_by`.
    """
    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    rows = []
    for run in runs:
        cfg = {f"config_{k}": v for k, v in run.config.items()}
        sm = {f"summary_{k}": v for k, v in run.summary.items()}
        rows.append({"run_id": run.id, **cfg, **sm})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found with the given filters")
        return df, None

    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(group_by_abbr, str):
        group_by_abbr = [group_by_abbr]
    abbr_map = (
        dict(zip(group_by, group_by_abbr))
        if group_by and group_by_abbr and len(group_by_abbr) == len(group_by)
        else {}
    )

    if group_by:
        cols = [f"config_{k}" for k in group_by if f"config_{k}" in df.columns]
        if not cols:
            print("No valid grouping columns found")
            return df, None

        metric_cols = [f"summary_{m}" for m in (metrics or [])]
        df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=metric_cols, inplace=True)

        for col in metric_cols:
            med = df[col].median()
            m = median_abs_deviation(df[col], scale="normal")
            if m:
                df = df[(df[col] - med).abs() <= m * mad_threshold]

        agg = {col: [aggregate, "std", "count"] for col in metric_cols}
        gdf = df.groupby(cols).agg(agg).reset_index()
        gdf.columns = ["_".join(filter(None, c)).strip("_") for c in gdf.columns]

        def make_label(row):
            parts = []
            for k, c in zip(group_by, cols):
                v = _abbreviate_val(row[c])
                parts.append(f"{abbr_map.get(k, k)}={v}")
            return " | ".join(parts)

        gdf["group_label"] = gdf.apply(make_label, axis=1)
        return df, gdf

    return df, None


def plot_grouped_metrics(
    grouped_df, metrics, show_variance=True, plot_type="bar", aggregate="mean"
):
    """Plot grouped metrics with optional variance—uses `group_label` for ticks"""
    if grouped_df is None or grouped_df.empty:
        print("No grouped data to plot")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for i, m in enumerate(metrics):
        mean_c = f"summary_{m}_{aggregate}"
        std_c = f"summary_{m}_std"
        cnt_c = f"summary_{m}_count"
        if mean_c not in grouped_df:
            print(f"Column {mean_c} not found")
            continue

        labels = grouped_df["group_label"]
        vals = grouped_df[mean_c]

        if plot_type == "bar":
            axes[i].bar(range(len(labels)), vals, alpha=0.7)
            if show_variance and std_c in grouped_df:
                errs = grouped_df[std_c] / np.sqrt(grouped_df[cnt_c])
                axes[i].errorbar(
                    range(len(labels)), vals, yerr=errs, fmt="none", capsize=5
                )
        else:  # scatter
            axes[i].scatter(range(len(labels)), vals, s=100, alpha=0.7)
            if show_variance and std_c in grouped_df:
                errs = grouped_df[std_c] / np.sqrt(grouped_df[cnt_c])
                axes[i].errorbar(
                    range(len(labels)), vals, yerr=errs, fmt="none", capsize=5
                )

        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45, ha="right")
        axes[i].set_title(f"{m}")
        axes[i].set_ylabel(f"{m} ({aggregate})")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main(
    entity="cruzas-universit-della-svizzera-italiana",
    project="tr_variants_assessment",
):
    filters = {
        "config.dataset_name": "mnist",
        "config.batch_size": 60000,
    }
    group_by_map = {
        "batch_size": "bs",
        "optimizer": "opt",
        "glob_second_order": "so",
        "glob_dogleg": "dleg",
    }

    # then, whenever you need the ordered lists:
    group_by = list(group_by_map.keys())
    group_by_abbr = list(group_by_map.values())
    metrics = ["accuracy", "loss"]

    df, gdf = analyze_wandb_runs_advanced(
        f"{entity}/{project}",
        filters=filters,
        group_by=group_by,
        group_by_abbr=group_by_abbr,
        metrics=metrics,
        show_variance=True,
        aggregate="mean",
        mad_threshold=3,
    )

    if gdf is not None:
        print(
            gdf[
                [
                    "group_label",
                    "summary_loss_mean",
                    "summary_loss_std",
                    "summary_loss_count",
                ]
            ]
        )
        plot_grouped_metrics(gdf, metrics, show_variance=True, plot_type="bar")


if __name__ == "__main__":
    main()
