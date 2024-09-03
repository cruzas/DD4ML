import os
import glob
import pandas as pd
from itertools import product


def compute_rowwise_statistics(file_list):
    """Compute row-wise mean and variance across multiple files."""
    dataframes = [pd.read_csv(file) for file in file_list]
    
    # For each data frame
    # - Add another column called "cum_time" which is the cumulative time
    for df in dataframes:
        df['cum_time'] = df['time'].cumsum()
        
    # Stack the dataframes to create a 3D array (rows, columns, files)
    stacked_data = pd.concat(dataframes, axis=0).groupby(level=0)
    
    # Compute row-wise mean and variance
    mean_df = stacked_data.mean()
    var_df = stacked_data.var()

    # Create a result dataframe with mean and variance columns
    result_df = pd.DataFrame({
        'loss_mean': mean_df['loss'],
        'loss_var': var_df['loss'],
        'accuracy_mean': mean_df['accuracy'],
        'accuracy_var': var_df['accuracy'],
        'time_mean': mean_df['cum_time'],
        'time_var': var_df['cum_time'],
    })
    
    return result_df

def speedup_statistics(file_list):
    # Compute the speedup statistics based on the cum_time measurements
    dataframes = [pd.read_csv(file) for file in file_list]

    


def main():
    # Identify all file patterns
    OPTIMIZERS = ["APTS"]
    LEARNING_RATES = [1.0]
    DATASETS = ["MNIST"]
    BATCH_SIZES = [10000]
    MODELS = ["feedforward"]
    NUM_SUBDOMAINS = [1]
    NUM_REPLICAS_PER_SUBDOMAIN = [2, 4, 8]
    NUM_STAGES_PER_REPLICA = [3]
    NUM_TRIALS = 3
    EPOCHS = 15

    # Nested for loop that will go through all combinations of the lists above
    for optimizer, lr, dataset, batch_size, model, num_subdomains, num_replicas_per_subdomain, num_stages_per_replica in product(
    OPTIMIZERS, LEARNING_RATES, DATASETS, BATCH_SIZES, MODELS, NUM_SUBDOMAINS, NUM_REPLICAS_PER_SUBDOMAIN, NUM_STAGES_PER_REPLICA):
    
        # Define base substring to look for
        base_substring = f"{optimizer}_{lr}_{dataset}_{batch_size}_{model}_{num_subdomains}_{num_replicas_per_subdomain}_{num_stages_per_replica}_{EPOCHS}"
        print("Looking for files with base substring:", base_substring)

        # Make a list containing all names of files that match the base substring
        filenames = [f for f in os.listdir(".") if base_substring in f]
        if len(filenames) < NUM_TRIALS:
            raise ValueError(f"Not enough files found for base substring {base_substring}")
        if len(filenames) > NUM_TRIALS:
            print(f"Warning: More files found for base substring {base_substring} than expected. Trimming down to {NUM_TRIALS} files.")
            # Sort filenames based on the *_tx.csv string
            filenames.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        aggregated_metrics = compute_rowwise_statistics(filenames)
        # Save the aggregated metrics to a csv file
        aggregated_metrics.to_csv(f"{base_substring}_mean_and_std.csv", index=False)

        

if __name__ == "__main__":
    main()