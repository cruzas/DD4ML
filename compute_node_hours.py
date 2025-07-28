import json

# Load the JSON data (replace 'your_file.json' with the actual file path)
with open("usage.json", "r") as f:
    data = json.load(f)

# Initialize total time
total_elapsed_seconds = 0

# Iterate through jobs
for job in data["jobs"]:
    # Add the elapsed time if available
    if "time" in job and "elapsed" in job["time"]:
        total_elapsed_seconds += job["time"]["elapsed"]

# Convert total seconds to hours:minutes:seconds
hours = total_elapsed_seconds // 3600
minutes = (total_elapsed_seconds % 3600) // 60
seconds = total_elapsed_seconds % 60

print(f"Total elapsed time: {hours}h {minutes}m {seconds}s")
