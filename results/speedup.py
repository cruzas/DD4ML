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

# Change working directory to the location of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def compute_speedup(base_substring, split_substring):
    # Get the list of all relevant files
    file_list = glob.glob(f"*{base_substring}.csv")
    # List file_list depending on the number right after splitting using split_substring
    file_list = sorted(file_list, key=lambda x: int(x.split(split_substring)[-1].split("_")[1]))

    print(f"File list: {file_list}")

    # Make a dataframe out of each file in file_list
    dataframes = [pd.read_csv(file) for file in file_list]
    
    # Create a new dataframe where we will have two columns. One is the number of replicas per subdomain and the other is the speedup
    speedup_dataframe = pd.DataFrame(columns=["num_copies", "speedup"])

    # Compute the speedup, where we compare the last entry in the time_mean column of the dataframe with the last entry in the time_mean column of the first dataframe
    for i in range(1, len(dataframes)):
        speedup = dataframes[0]['time_mean'].iloc[-1] / dataframes[i]['time_mean'].iloc[-1]
        num_copies = int(file_list[i].split(split_substring)[-1].split("_")[1])
        print(f"Speedup for {num_copies} replicas per subdomain: {speedup}")
        speedup_dataframe = speedup_dataframe.append({"num_copies": num_copies, "speedup": speedup}, ignore_index=True)

    # Save the speedup dataframe to a csv file
    speedup_dataframe.to_csv(f"speedup_{base_substring.replace('_mean_and_std', '')}.csv", index=False)

def main():
    # Define the base substring pattern
    NUM_SUBDOMAINS = 1
    # NUM_REPLICAS_PER_SUBDOMAIN is what varies here
    base_substring = f"{OPTIMIZER}_{LR}_{DATASET}_{BATCH_SIZE}_{MODEL}_{NUM_SUBDOMAINS}_*_{NUM_STAGES_PER_REPLICA}_{EPOCHS}_mean_and_std"
    split_substring = f"{OPTIMIZER}_{LR}_{DATASET}_{BATCH_SIZE}_{MODEL}_{NUM_SUBDOMAINS}"
    # Compute the speedup statistics
    compute_speedup(base_substring, split_substring)

    # Define the base substring pattern
    NUM_REPLICAS_PER_SUBDOMAIN = 1
    # NUM_SUBDOMAINS is what varies here
    base_substring = f"{OPTIMIZER}_{LR}_{DATASET}_{BATCH_SIZE}_{MODEL}_*_{NUM_REPLICAS_PER_SUBDOMAIN}_{NUM_STAGES_PER_REPLICA}_{EPOCHS}_mean_and_std"
    split_substring = f"{OPTIMIZER}_{LR}_{DATASET}_{BATCH_SIZE}_{MODEL}"
    # Compute the speedup statistics
    compute_speedup(base_substring, split_substring)


if __name__ == "__main__":
    main()