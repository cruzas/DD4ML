#!/bin/bash

# Set environment variables
export JOB_NAME="run_config_file"
export SCRIPT="${JOB_NAME}.py"
export USE_WANDB=1  # Set to 1 if needed
export NCCL_DEBUG=WARN  # Adjust as needed

# Get the current working directory
current_dir=$(pwd)

# Check whether current_dir has "home" as a substring
if [[ "$current_dir" == *"home"* ]]; then
    echo "Submitting job to USI Rosa..."
    envsubst < job_template_rosa.job | sbatch
else
    echo "Submitting job to Daint ALPS..."
    envsubst < job_template_daintalps.job | sbatch
fi
