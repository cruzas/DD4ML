import matplotlib.pyplot as plt
import pandas as pd

# Load the data files
files = {
    "SGD, nsd=2": "../results/tshakespeare_t_0_nsd_2_nrs_1_nst_13_bs_2048_parsgd_iter.csv",
    "APTS, nsd=2": "../results/tshakespeare_t_0_nsd_2_nrs_1_nst_13_bs_2048_iter.csv",
}

# Define plot styles
styles = {
    "SGD, nsd=2": {"color": "blue", "linestyle": "-"},
    "APTS, nsd=2": {"color": "blue", "linestyle": "--"},
}

# Plot the data
plt.figure(figsize=(10, 6))
for label, filepath in files.items():
    data = pd.read_csv(filepath)
    plt.plot(data["iteration"], data["loss"], label=label, **styles[label])

# Customize the plot
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs Iteration")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
