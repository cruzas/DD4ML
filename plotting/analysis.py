import pandas as pd
from scipy.stats import median_abs_deviation

import wandb

from .utils import _abbreviate_val


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
        pprint.pprint("No runs found with the given filters")
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
            pprint.pprint("No valid grouping columns found")
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
