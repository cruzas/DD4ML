#!/bin/bash

# Strong Scaling Test Script

RUNPATH=/scratch/snx3000/scruzale/ML_APTS/
cd $RUNPATH || exit

# Fixed parameters
MODELS=("feedforward")
OPTIMIZERS=("APTS")
DATASETS=("MNIST")
BATCH_SIZES=(28000)
NUM_SUBDOMAINS=(2)
NUM_REPLICAS_PER_SUBDOMAIN=(1)
NUM_STAGES_PER_REPLICA=(3)
EPOCHS=5
NUM_TRIALS=1

# Output log directory
LOG_DIR="./scaling_logs/strong_scaling_logs"

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

    # If the file $optimizer_$dataset_$batch_size_$model_$num_subdomains_$num_replicas_per_subdomain_$num_stages_per_replica_t$trial.out exists, increase trial by one
    while [ -f "$LOG_DIR/${optimizer}_${dataset}_${batch_size}_${model}_${num_subdomains}_${num_replicas_per_subdomain}_${num_stages_per_replica}_t${trial}.out" ]; do
        trial=$((trial + 1))
    done

    job_name="${optimizer}_${dataset}_${batch_size}_${model}_${num_subdomains}_${num_replicas_per_subdomain}_${num_stages_per_replica}_${EPOCHS}_t${trial}"
    error_file="$LOG_DIR/${job_name}.err"
    output_file="$LOG_DIR/${job_name}.out"

    nodes=$((num_subdomains * num_replicas_per_subdomain * num_stages_per_replica))

    sbatch --nodes="$nodes" --job-name="$job_name" --output="$output_file" --error="$error_file" cluster_scripts/strong_scalability.job "$optimizer" "$dataset" "$batch_size" "$model" "$num_subdomains" "$num_replicas_per_subdomain" "$num_stages_per_replica" "$trial" "$EPOCHS"
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

