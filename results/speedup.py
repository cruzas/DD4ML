import pandas as pd
import glob
import os

# Define the parameters for the base substring
OPTIMIZER = "APTS"
LR = 1.0
DATASET = "MNIST"
BATCH_SIZE = "10000"
MODEL = "feedforward"
NUM_STAGES_PER_REPLICA = 3
NUM_TRIALS = 3
EPOCHS = 15

def compute_speedup(base_substring):
    # Get the list of all relevant files
    file_list = glob.glob(f"*{base_substring}.csv")
    file_list.sort()

    print(f"File list: {file_list}")

    # Make a dataframe out of each file in file_list
    dataframes = [pd.read_csv(file) for file in file_list]
    
    # Create a new dataframe where we will have two columns. One is the number of replicas per subdomain and the other is the speedup
    speedup_dataframes = pd.DataFrame(columns=["replicas_per_subdomain", "speedup"])

    # Compute the speedup, where we compare the last entry in the time_mean column of the dataframe with the last entry in the time_mean column of the first dataframe
    for i in range(1, len(dataframes)):
        speedup = dataframes[0].loc[dataframes[0].index[-1], "time_mean"] / dataframes[i].loc[dataframes[i].index[-1], "time_mean"]
        replicas_per_subdomain = int(file_list[i].split("_")[-2])
        print(f"Speedup for {replicas_per_subdomain} replicas per subdomain: {speedup}")
        speedup_dataframes = speedup_dataframes.append({"replicas_per_subdomain": replicas_per_subdomain, "speedup": speedup}, ignore_index=True)

    print(speedup_dataframes)

def main():
    # Define the base substring pattern
    NUM_SUBDOMAINS = 1
    # NUM_REPLICAS_PER_SUBDOMAIN is what varies here
    base_substring = f"{OPTIMIZER}_{LR}_{DATASET}_{BATCH_SIZE}_{MODEL}_{NUM_SUBDOMAINS}_*_{NUM_STAGES_PER_REPLICA}_{EPOCHS}_mean_and_std"

    # Compute the speedup statistics
    compute_speedup(base_substring)

if __name__ == "__main__":
    main()