#!/bin/bash
export JOB_NAME="run_config_file"
export SCRIPT="${JOB_NAME}.py"
export USE_WANDB=1  # Set to 1 if needed
export NCCL_DEBUG=WARN # Set to INFO if needed

envsubst < job_template.job | sbatch