#!/bin/bash

# Define parameter arrays
NUM_STAGES_ARR=(2)
NUM_SUBD_ARR=(1)
NUM_REP_ARR=(1)

# Get current working directory
current_dir=$(pwd)

# Iterate over all combinations
for NUM_STAGES in "${NUM_STAGES_ARR[@]}"; do
    for NUM_SUBD in "${NUM_SUBD_ARR[@]}"; do
        for NUM_REP in "${NUM_REP_ARR[@]}"; do
            # Calculate nodes
            nodes=$(( (NUM_STAGES * NUM_SUBD * NUM_REP + 3) / 4 )) # Rounds up in case division is not exact
            # Define job name and environment variables
            JOB_NAME="apts_nst_${NUM_STAGES}_nsd_${NUM_SUBD}_nrpsd_${NUM_REP}"
            export JOB_NAME
            export SCRIPT="run_config_file.py"
            export USE_WANDB=1
            export NCCL_DEBUG=WARN
            export NUM_STAGES
            export NUM_SUBD
            export NUM_REP

            if [[ "$current_dir" == *"home"* ]]; then
                echo "Submitting job ${JOB_NAME} to USI Rosa with ${nodes} nodes..."
                envsubst < apts_rosa.job | sbatch --nodes=${nodes}
            else
                echo "Submitting job ${JOB_NAME} to Daint ALPS with ${nodes} nodes..."
                envsubst < apts_daintalps.job | sbatch --nodes=${nodes}
            fi
        done
    done
done