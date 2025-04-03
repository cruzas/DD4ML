#!/bin/bash

# Define parameter arrays
NUM_STAGES_ARR=(1)
NUM_SUBD_ARR=(4 8)
NUM_REP_ARR=(1)
BATCH_SIZES=(128)
OPTIMIZER="apts_d"
DATASET="cifar10"
MODEL="simple_resnet"
CRITERION="cross_entropy"
EPOCHS=10
GLOBAL_PASS=True
FOC=False
GRADIENT_ACCUMULATION=True
ACCUMULATION_STEPS=3
TRIALS=1

# Get current working directory
current_dir=$(pwd)
# System maximum GPUs per node (used as an upper bound)
if [[ "$current_dir" == *"home"* ]]; then
    max_ngpu_per_node=1 # 2 only for multi-gpu on USI Rosa
else
    max_ngpu_per_node=4 # each node has 4 GPUs on Daint Alps
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
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                for TRIAL in $(seq 1 "${TRIALS}"); do
                    JOB_NAME="${OPTIMIZER}_${DATASET}_${BATCH_SIZE}_nst_${NUM_STAGES}_nsd_${NUM_SUBD}_nrpsd_${NUM_REP}_trial_${TRIAL}"
                    WORLD_SIZE=$((NUM_STAGES * NUM_SUBD * NUM_REP))

                    echo "World size: ${WORLD_SIZE}"
                    nodes=$(calc_nodes)
                    echo "Requesting ${nodes} nodes for homogeneous task distribution."
                    echo "Number of nodes to allocate: ${nodes}"

                    # Compute tasks per node dynamically
                    NTASKS_PER_NODE=$((WORLD_SIZE / nodes))
                    echo "Tasks per node: ${NTASKS_PER_NODE}"

                    # Create filename
                    CONFIG_FILE="./config_files/config_${JOB_NAME}.yaml"

                    # Copy the base config
                    cp ./config_files/config_${OPTIMIZER}.yaml "${CONFIG_FILE}"

                    # Update the values in the copied file
                    sed -i '/num_stages:/ {n; s/value: .*/value: '"${NUM_STAGES}"'/}' "${CONFIG_FILE}"
                    sed -i '/num_subdomains:/ {n; s/value: .*/value: '"${NUM_SUBD}"'/}' "${CONFIG_FILE}"
                    sed -i '/num_replicas_per_subdomain:/ {n; s/value: .*/value: '"${NUM_REP}"'/}' "${CONFIG_FILE}"
                    sed -i '/dataset_name:/ {n; s/value: .*/value: '"${DATASET}"'/}' "${CONFIG_FILE}"
                    sed -i '/model_name:/ {n; s/value: .*/value: '"${MODEL}"'/}' "${CONFIG_FILE}"
                    sed -i '/criterion:/ {n; s/value: .*/value: '"${CRITERION}"'/}' "${CONFIG_FILE}"
                    sed -i '/epochs:/ {n; s/value: .*/value: '"${EPOCHS}"'/}' "${CONFIG_FILE}"
                    sed -i '/gradient_accumulation:/ {n; s/value: .*/value: '"${GRADIENT_ACCUMULATION}"'/}' "${CONFIG_FILE}"
                    if [[ "$OPTIMIZER" == "apts" ]]; then
                        sed -i '/batch_size:/ {n; s/value: .*/value: '"${BATCH_SIZE}"'/}' "${CONFIG_FILE}"
                    else
                        effective_batch_size=${BATCH_SIZE}
                        actual_batch_size=$((effective_batch_size * NUM_SUBD))
                        echo "Effective batch size: ${effective_batch_size}"
                        echo "Actual batch size: ${actual_batch_size}"
                        sed -i '/batch_size:/ {n; s/value: .*/value: '"${actual_batch_size}"'/}' "${CONFIG_FILE}"
                        sed -i '/effective_batch_size:/ {n; s/value: .*/value: '"${effective_batch_size}"'/}' "${CONFIG_FILE}"
                    fi

                    if [[ "$OPTIMIZER" == "apts_d" ]]; then
                        sed -i '/global_pass:/ {n; s/value: .*/value: '"${GLOBAL_PASS}"'/}' "${CONFIG_FILE}"
                        sed -i '/foc:/ {n; s/value: .*/value: '"${FOC}"'/}' "${CONFIG_FILE}"
                    fi

                    export JOB_NAME SCRIPT="run_config_file.py" USE_WANDB=1 NCCL_DEBUG=WARN
                    export NUM_STAGES NUM_SUBD NUM_REP WORLD_SIZE tasks_remaining=${WORLD_SIZE} NTASKS_PER_NODE CONFIG_FILE OPTIMIZER TRIAL

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
    done
done
