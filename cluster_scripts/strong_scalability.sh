#!/bin/bash

# Strong Scaling Test Script

# Fixed parameters
MODELS=("NN")
OPTIMIZERS=("APTS")
DATASETS=("MNIST")
BATCH_SIZES=(32 64 128)
NUM_SUBDOMAINS=(2 4 8)
NUM_REPLICAS_PER_SUBDOMAIN=(1)
NUM_STAGES_PER_REPLICA=(2 4 8)
EPOCHS=2
NUM_TRIALS=2

# Output log directory
LOG_DIR="../scaling_logs/strong_scaling_logs"

# Check if LOG_DIR exists. If not, create it.
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi

submit_job() {
    local optimizer=$1
    local dataset=$2
    local batch_size=$3
    local model=$4
    local num_subdomains=$5
    local num_replicas_per_subdomain=$6
    local num_stages_per_replica=$7
    local trial=$8

    job_name="${optimizer}_${dataset}_${batch_size}_${model}_${num_subdomains}_${num_replicas_per_subdomain}_${num_stages_per_replica}_t${trial}"
    error_file="${job_name}.err"
    output_file="${job_name}.out"

    # sbatch --job-name="$job_name" --output="$output_file" --error="$error_file" strong_scaling.job "$optimizer" "$dataset" "$batch_size" "$model" "$num_subdomains" "$num_replicas_per_subdomain" "$num_stages_per_replica" "$trial" "$EPOCHS"
    python3 -u ../tests/scaling_test.py --optimizer "$optimizer" --dataset "$dataset" --batch_size "$batch_size" --model "$model" --num_subdomains "$num_subdomains" --num_replicas_per_subdomain "$num_replicas_per_subdomain" --num_stages_per_replica "$num_stages_per_replica" --epochs "$EPOCHS" --trial "$trial"
}


for optimizer in "${OPTIMIZERS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        for batch_size in "${BATCH_SIZES[@]}"
        do
            for model in "${MODELS[@]}"
            do
                for num_subdomains in "${NUM_SUBDOMAINS[@]}"
                do
                    for num_replicas_per_subdomain in "${NUM_REPLICAS_PER_SUBDOMAIN[@]}"
                    do
                        for num_stages_per_replica in "${NUM_STAGES_PER_REPLICA[@]}"
                        do
                            for trial in $(seq 0 $(($NUM_TRIALS - 1)))
                            do 
                                echo "Running $optimizer on $dataset with batch size $batch_size, model $model, $num_subdomains subdomains, $num_replicas_per_subdomain replicas per subdomain, $num_stages_per_replica stages per replica, trial $trial" for $EPOCHS epochs
                                submit_job "$optimizer" "$dataset" "$batch_size" "$model" "$num_subdomains" "$num_replicas_per_subdomain" "$num_stages_per_replica" "$trial"
                            done
                        done
                    done
                done
            done
        done
    done
done

