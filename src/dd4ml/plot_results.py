import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

import wandb


def plot_averaged_time_series(
    project_path,
    filters=None,
    group_by=None,
    metric_name="train_loss",
    show_variance=True,
):
    """Plot time series data averaged across runs with same configuration"""

    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    if isinstance(group_by, str):
        group_by = [group_by]

    # Group runs by configuration
    grouped_runs = {}
    for run in runs:
        if group_by:
            # Create group key from config values
            group_key = tuple(run.config.get(key, "unknown") for key in group_by)
            group_label = " | ".join(
                [f"{key}={val}" for key, val in zip(group_by, group_key)]
            )
        else:
            group_key = "all_runs"
            group_label = "All Runs"

        if group_key not in grouped_runs:
            grouped_runs[group_key] = {"runs": [], "label": group_label}
        grouped_runs[group_key]["runs"].append(run)

    plt.figure(figsize=(12, 8))

    for group_key, group_info in grouped_runs.items():
        runs_in_group = group_info["runs"]
        label = group_info["label"]

        # Get all histories for this group
        histories = []
        for run in runs_in_group:
            history = run.history()
            if metric_name in history.columns:
                histories.append(history[["_step", metric_name]].dropna())

        if not histories:
            continue

        # Find common step range
        min_steps = min(h["_step"].min() for h in histories)
        max_steps = max(h["_step"].max() for h in histories)

        # Create common step grid
        step_grid = np.linspace(min_steps, max_steps, 100)

        # Interpolate all runs to common grid
        interpolated_values = []
        for history in histories:
            interpolated = np.interp(step_grid, history["_step"], history[metric_name])
            interpolated_values.append(interpolated)

        # Calculate mean and std
        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)

        # Plot mean
        plt.plot(
            step_grid, mean_values, label=f"{label} (n={len(histories)})", linewidth=2
        )

        # Plot variance as shaded area
        if show_variance and len(histories) > 1:
            plt.fill_between(
                step_grid, mean_values - std_values, mean_values + std_values, alpha=0.2
            )

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Time (Averaged by Configuration)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


from scipy.stats import median_abs_deviation


def analyze_wandb_runs_advanced(
    project_path,
    filters=None,
    group_by=None,
    metrics=None,
    plot_type="scatter",
    show_variance=True,
    aggregate="mean",
    mad_threshold=3,
):
    """
    Analyze wandb runs with multi-key grouping, robust filtering, and variance analysis.

    Args:
        project_path: "username/project-name"
        filters: Dict of filters to apply
        group_by: String or list of config parameters to group by
        metrics: List of metrics to plot (e.g. ["accuracy", "train_loss"])
        plot_type: 'scatter', 'box', 'line', 'bar'
        show_variance: Whether to show error bars
        aggregate: 'mean', 'median', 'max', 'min'
        mad_threshold: threshold (in MADs) for filtering outliers
    """
    api = wandb.Api()
    runs = api.runs(project_path, filters=filters or {})

    data = []
    for run in runs:
        config = {f"config_{k}": v for k, v in run.config.items()}
        summary = {f"summary_{k}": v for k, v in run.summary.items()}
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            **config,
            **summary,
        }
        data.append(row)

    df = pd.DataFrame(data)

    if df.empty:
        print("No runs found with the given filters")
        return df, None

    if isinstance(group_by, str):
        group_by = [group_by]

    if group_by:
        group_cols = [f"config_{key}" for key in group_by]

        # Check valid grouping keys
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            group_cols = [col for col in group_cols if col in df.columns]

        if not group_cols:
            print("No valid grouping columns found")
            return df, None

        # Ensure metric columns exist and are numeric
        metric_cols = [f"summary_{m}" for m in metrics or []]
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=metric_cols, inplace=True)

        # Remove outliers using robust z-score (MAD) per metric
        for col in metric_cols:
            median = df[col].median()
            mad = median_abs_deviation(df[col], scale="normal")
            if mad == 0:
                continue  # skip if constant
            z = abs(df[col] - median) / mad
            df = df[z <= mad_threshold]

        # Aggregate
        agg_funcs = {f"summary_{m}": [aggregate, "std", "count"] for m in metrics}
        grouped_df = df.groupby(group_cols).agg(agg_funcs).reset_index()

        # Flatten column names
        grouped_df.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in grouped_df.columns.values
        ]

        # Create group label
        if len(group_cols) == 1:
            grouped_df["group_label"] = grouped_df[group_cols[0]].astype(str)
        else:
            grouped_df["group_label"] = grouped_df[group_cols].apply(
                lambda row: " | ".join(
                    f"{col.replace('config_', '')}={val}"
                    for col, val in zip(group_cols, row)
                ),
                axis=1,
            )

        return df, grouped_df

    return df, None


def plot_grouped_metrics(
    grouped_df, metrics, show_variance=True, plot_type="bar", aggregate="mean"
):
    """Plot grouped metrics with optional variance"""

    if grouped_df is None or grouped_df.empty:
        print("No grouped data to plot")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(8 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        mean_col = f"summary_{metric}_{aggregate}"
        std_col = f"summary_{metric}_std"
        count_col = f"summary_{metric}_count"

        if mean_col not in grouped_df.columns:
            print(f"Column {mean_col} not found")
            continue

        x_labels = grouped_df["group_label"]
        y_values = grouped_df[mean_col]

        if plot_type == "bar":
            bars = axes[i].bar(range(len(x_labels)), y_values, alpha=0.7)

            if show_variance and std_col in grouped_df.columns:
                # Calculate standard error
                counts = grouped_df[count_col] if count_col in grouped_df.columns else 1
                std_err = grouped_df[std_col] / np.sqrt(counts)
                axes[i].errorbar(
                    range(len(x_labels)),
                    y_values,
                    yerr=std_err,
                    fmt="none",
                    color="black",
                    capsize=5,
                )

            axes[i].set_xticks(range(len(x_labels)))
            axes[i].set_xticklabels(x_labels, rotation=45, ha="right")

        elif plot_type == "scatter":
            axes[i].scatter(range(len(x_labels)), y_values, alpha=0.7, s=100)

            if show_variance and std_col in grouped_df.columns:
                counts = grouped_df[count_col] if count_col in grouped_df.columns else 1
                std_err = grouped_df[std_col] / np.sqrt(counts)
                axes[i].errorbar(
                    range(len(x_labels)),
                    y_values,
                    yerr=std_err,
                    fmt="none",
                    color="black",
                    capsize=5,
                )

            axes[i].set_xticks(range(len(x_labels)))
            axes[i].set_xticklabels(x_labels, rotation=45, ha="right")

        axes[i].set_ylabel(f"{metric} ({aggregate})")
        axes[i].set_title(f"{metric} by Groups")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage with multiple grouping keys
def main(
    entity="cruzas-universit-della-svizzera-italiana", project="tr_variants_assessment"
):
    filters = {
        "config.optimizer": "tr",  # string
        "config.dataset_name": "mnist",  # string
        "config.batch_size": 60000,  # int
        # "config.glob_second_order": True,  # bool
        # "config.glob_dogleg": True,  # bool
    }

    # Group by multiple keys
    group_by = ["glob_second_order", "glob_dogleg"]
    group_by_abbr = ["so", "dleg"]
    metrics = ["accuracy", "loss"]

    df, grouped_df = analyze_wandb_runs_advanced(
        f"{entity}/{project}",
        filters=filters,
        group_by=group_by,
        metrics=metrics,
        show_variance=True,
        aggregate="mean",
        mad_threshold=3,  # adjustable sensitivity
    )

    print(
        grouped_df[
            [
                "group_label",
                "summary_loss_mean",
                "summary_loss_std",
                "summary_loss_count",
            ]
        ]
    )

    if grouped_df is not None:
        print("Grouped Statistics:")
        print(grouped_df)

        # Plot the results
        plot_grouped_metrics(grouped_df, metrics, show_variance=True, plot_type="bar")

        # You can also access the raw statistics
        for metric in metrics:
            mean_col = f"summary_{metric}_mean"
            std_col = f"summary_{metric}_std"
            count_col = f"summary_{metric}_count"

            if all(col in grouped_df.columns for col in [mean_col, std_col, count_col]):
                print(f"\n{metric} Statistics:")
                for _, row in grouped_df.iterrows():
                    print(
                        f"  {row['group_label']}: "
                        f"{row[mean_col]:.4f} ± {row[std_col]:.4f} "
                        f"(n={row[count_col]})"
                    )


if __name__ == "__main__":
    main()
