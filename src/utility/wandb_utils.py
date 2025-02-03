import pandas as pd
import wandb


def compute_best_lr_per_batch_size(project, metric="loss", verbose=False):
    print("Fetching run data...")
    records = fetch_run_data(project)
    
    if not records:
        print("No records found.")
        return

    per_epoch, overall = compute_metrics(records)
    
    best_lr = best_learning_rates(overall, metric=metric)
    
    print(f"Best learning rate per batch size based on average {metric}:")
    print(best_lr[["batch_size", "learning_rate", metric]])
    
    if verbose:
        print("\nAveraged metrics per epoch for each (batch_size, learning_rate) combination:")
        print(per_epoch)
    

def fetch_run_data(project):
    """Fetch run history data from W&B for the given project.

    Args:
        project (str): Project identifier (e.g., "entity/project_name").

    Returns:
        list of dict: Records with batch_size, learning_rate, epoch, loss, accuracy, and running_time.
    """
    api = wandb.Api()
    runs = api.runs(project)
    records = []

    for run in runs:
        config = run.config
        lr = config.get("learning_rate")
        bs = config.get("batch_size")
        history = run.history(keys=["epoch", "loss", "accuracy", "running_time"])
        if history.empty:
            continue
        for _, row in history.iterrows():
            records.append({
                "batch_size": bs,
                "learning_rate": lr,
                "epoch": row["epoch"],
                "loss": row["loss"],
                "accuracy": row["accuracy"],
                "running_time": row["running_time"]
            })
    return records

def compute_metrics(records):
    """Compute average metrics per epoch and overall averages.

    Args:
        records (list of dict): List of records from W&B runs.

    Returns:
        tuple: A DataFrame with per-epoch averages and a DataFrame with overall averages.
    """
    df = pd.DataFrame(records)
    grouped = df.groupby(["batch_size", "learning_rate", "epoch"], as_index=False).mean()
    overall = grouped.groupby(["batch_size", "learning_rate"], as_index=False).agg({
        "loss": "mean",
        "accuracy": "mean",
        "running_time": "mean"
    })
    return grouped, overall

def best_learning_rates(overall, metric="loss"):
    """
    Determine the best learning rate for each batch size based on the chosen metric.
    
    Args:
        overall (pd.DataFrame): DataFrame containing overall metrics for each (batch_size, learning_rate) combination.
        metric (str): Metric to base the evaluation on; options are "loss" or "accuracy".
        
    Returns:
        pd.DataFrame: Best learning rates per batch size.
    """
    if metric == "loss":
        best_lr = overall.loc[overall.groupby("batch_size")["loss"].idxmin()]
    elif metric == "accuracy":
        best_lr = overall.loc[overall.groupby("batch_size")["accuracy"].idxmax()]
    else:
        raise ValueError("Unsupported metric. Use 'loss' or 'accuracy'.")
    
    return best_lr