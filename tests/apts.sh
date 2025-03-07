#!/bin/bash

# Define parameter arrays
NUM_STAGES_ARR=(6)
NUM_SUBD_ARR=(1)
NUM_REP_ARR=(1)

# Get current working directory
current_dir=$(pwd)
ngpu_per_node=4  # Adapt to your system

# Helper function for job submission
submit_job() {
    local template="$1"
    local temp_job
    temp_job=$(mktemp)
    sed -e "s|\${JOB_NAME}|${JOB_NAME}|g" \
        -e "s|\${WORLD_SIZE}|${WORLD_SIZE}|g" \
        -e "s|\${SCRIPT}|${SCRIPT}|g" \
        -e "s|\${NUM_STAGES}|${NUM_STAGES}|g" \
        -e "s|\${NUM_SUBD}|${NUM_SUBD}|g" \
        -e "s|\${NUM_REP}|${NUM_REP}|g" \
        "${template}" > "${temp_job}"
    sbatch --nodes=${nodes} "${temp_job}"
    rm "${temp_job}"
}

# Iterate over all combinations
for NUM_STAGES in "${NUM_STAGES_ARR[@]}"; do
    for NUM_SUBD in "${NUM_SUBD_ARR[@]}"; do
        for NUM_REP in "${NUM_REP_ARR[@]}"; do
            JOB_NAME="apts_nst_${NUM_STAGES}_nsd_${NUM_SUBD}_nrpsd_${NUM_REP}"
            WORLD_SIZE=$((NUM_STAGES * NUM_SUBD * NUM_REP))
            echo "World size: ${WORLD_SIZE}"
            nodes=$(((WORLD_SIZE + ngpu_per_node - 1) / ngpu_per_node))
            echo "Number of nodes to allocate: ${nodes}"
        
            export JOB_NAME SCRIPT="run_config_file.py" USE_WANDB=1 NCCL_DEBUG=WARN
            export NUM_STAGES NUM_SUBD NUM_REP WORLD_SIZE tasks_remaining=${WORLD_SIZE}

            if [[ "$current_dir" == *"home"* ]]; then
                echo "Submitting job ${JOB_NAME} to USI Rosa with ${nodes} nodes..."
                submit_job apts_rosa.job
            else
                echo "Submitting job ${JOB_NAME} to Daint ALPS with ${nodes} nodes..."
                submit_job apts_daintalps.job
            fi
        done
    done
done
