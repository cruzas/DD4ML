import pandas as pd
import wandb

# Initialize wandb API
api = wandb.Api()

# Define your project name
PROJECT_NAME = "your_entity/your_project"  # Replace with your wandb entity/project

# Fetch all runs
runs = api.runs(PROJECT_NAME)

# Extract relevant data
records = []
for run in runs:
    summary = run.summary
    config = run.config
    
    records.append({
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "optimizer": config.get("optimizer"),
        "dataset_name": config.get("dataset_name"),
        "model_name": config.get("model_name"),
        "criterion": config.get("criterion"),
        "loss": summary.get("loss"),
        "accuracy": summary.get("accuracy")
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# Compute average loss and accuracy per hyperparameter combination
agg_df = df.groupby(["learning_rate", "batch_size", "epochs", "optimizer", "dataset_name", "model_name", "criterion"])
agg_df = agg_df.agg({"loss": "mean", "accuracy": "mean"}).reset_index()

# Find best learning rate per batch size based on lowest loss and highest accuracy
best_by_loss = agg_df.loc[agg_df.groupby("batch_size")["loss"].idxmin(), ["batch_size", "learning_rate", "loss"]]
best_by_accuracy = agg_df.loc[agg_df.groupby("batch_size")["accuracy"].idxmax(), ["batch_size", "learning_rate", "accuracy"]]

# Display results
print("Aggregated Results")
print(agg_df)

print("Best Learning Rate by Loss")
print(best_by_loss)

print("Best Learning Rate by Accuracy")
print(best_by_accuracy)
