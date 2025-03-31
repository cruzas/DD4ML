#!/bin/bash

# Define parameter arrays
NUM_STAGES_ARR=(1)
NUM_SUBD_ARR=(4)
NUM_REP_ARR=(1)
BATCH_SIZE=50000
OPTIMIZER="apts_d"
DATASET="cifar10"

# Get current working directory
current_dir=$(pwd)
# System maximum GPUs per node (used as an upper bound)
if [[ "$current_dir" == *"home"* ]]; then
    max_ngpu_per_node=2
else
    max_ngpu_per_node=4
fi

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
        -e "s|\${NTASKS_PER_NODE}|${NTASKS_PER_NODE}|g" \
        "${template}" >"${temp_job}"
    sbatch --nodes=${nodes} "${temp_job}"
    rm "${temp_job}"
}

# Determine homogeneous node allocation.
calc_nodes() {
    for n in $(seq 1 $WORLD_SIZE); do
        if [ $((WORLD_SIZE % n)) -eq 0 ]; then
            tasks_per_node=$((WORLD_SIZE / n))
            if [ $tasks_per_node -le $max_ngpu_per_node ]; then
                echo "$n"
                return
            fi
        fi
    done
    # Fallback: one task per node
    echo "$WORLD_SIZE"
}

# Iterate over all combinations
for NUM_STAGES in "${NUM_STAGES_ARR[@]}"; do
    for NUM_SUBD in "${NUM_SUBD_ARR[@]}"; do
        for NUM_REP in "${NUM_REP_ARR[@]}"; do
            JOB_NAME="${OPTIMIZER}_${DATASET}_${BATCH_SIZE}_nst_${NUM_STAGES}_nsd_${NUM_SUBD}_nrpsd_${NUM_REP}"
            WORLD_SIZE=$((NUM_STAGES * NUM_SUBD * NUM_REP))

            echo "World size: ${WORLD_SIZE}"
            nodes=$(calc_nodes)
            echo "Requesting ${nodes} nodes for homogeneous task distribution."
            echo "Number of nodes to allocate: ${nodes}"

            # Compute tasks per node dynamically
            NTASKS_PER_NODE=$((WORLD_SIZE / nodes))
            echo "Tasks per node: ${NTASKS_PER_NODE}"

            # Create filename
            CONFIG_FILE="./config_files/config_${OPTIMIZER}_nst_${NUM_STAGES}_nsd_${NUM_SUBD}_nrpsd_${NUM_REP}.yaml"

            # Copy the base config
            cp ./config_files/config_${OPTIMIZER}.yaml "${CONFIG_FILE}"

            # Update the values in the copied file
            sed -i '/num_stages:/ {n; s/value: .*/value: '"${NUM_STAGES}"'/}' "${CONFIG_FILE}"
            sed -i '/num_subdomains:/ {n; s/value: .*/value: '"${NUM_SUBD}"'/}' "${CONFIG_FILE}"
            sed -i '/num_replicas_per_subdomain:/ {n; s/value: .*/value: '"${NUM_REP}"'/}' "${CONFIG_FILE}"
            sed -i '/dataset:/ {n; s/value: .*/value: '"${DATASET}"'/}' "${CONFIG_FILE}"
            sed -i '/batch_size:/ {n; s/value: .*/value: '"${BATCH_SIZE}"'/}' "${CONFIG_FILE}"

            export JOB_NAME SCRIPT="run_config_file.py" USE_WANDB=1 NCCL_DEBUG=WARN
            export NUM_STAGES NUM_SUBD NUM_REP WORLD_SIZE tasks_remaining=${WORLD_SIZE} NTASKS_PER_NODE CONFIG_FILE OPTIMIZER

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
