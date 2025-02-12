#!/bin/bash
export JOB_NAME="example"
export SCRIPT="${JOB_NAME}.py"
export USE_WANDB=0  # Set to 1 if needed

envsubst < job_template.job | sbatch