#!/bin/bash

RUNPATH=/scratch/snx3000/scruzale/ML_APTS/
cd $RUNPATH || exit

# Fixed parameters
NUM_EPOCHS=2
DATA_CHUNKS_AMOUNT=10
BLOCK_SIZE=256
VOCAB_SIZE=0
N_LAYER=2
N_HEAD=2
N_EMBD=384
DROPOUT=0.0
LEARNING_RATE=0.001
PERCENTAGE=50.0

# Arrays of parameters that change
TRIALS=(0)
NUM_SUBDOMAINS_LIST=(2)
NUM_REPLICAS_PER_SUBDOMAIN_LIST=(1)
NUM_STAGES_LIST=(13)
BATCH_SIZES=(2048)

LOG_DIR="./results"

# Check if LOG_DIR exists. If not, create it.
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

submit_job() {
    local trial=$1
    local num_epochs=$2
    local num_subdomains=$3
    local num_replicas_per_subdomain=$4
    local num_stages=$5
    local seed=$6
    local batch_size=$7
    local data_chunks_amount=$8
    local block_size=$9
    local vocab_size=${10}
    local n_layer=${11}
    local n_head=${12}
    local n_embd=${13}
    local dropout=${14}
    local learning_rate=${15}
    local percentage=${16}

    echo "Submitting job with trial=$trial, num_epochs=$num_epochs, num_subdomains=$num_subdomains, num_replicas_per_subdomain=$num_replicas_per_subdomain, num_stages=$num_stages, seed=$seed, batch_size=$batch_size, data_chunks_amount=$data_chunks_amount, block_size=$block_size, vocab_size=$vocab_size, n_layer=$n_layer, n_head=$n_head, n_embd=$n_embd, dropout=$dropout, learning_rate=$learning_rate, percentage=$percentage"

    # Make a string replacing lr "." with "_"
    lr_str=$(echo "$learning_rate" | tr . _)
    perc_str=$(echo "$percentage" | tr . _)

    job_name="t_${trial}_ns_${num_subdomains}_nr_${num_replicas_per_subdomain}_st_${num_stages}_bs_${batch_size}_lr_${lr_str}_perc_${perc_str}_parsgd"
    error_file="$LOG_DIR/${job_name}.err"
    output_file="$LOG_DIR/${job_name}.out"

    # Calculate number of nodes
    nodes=$((num_subdomains * num_replicas_per_subdomain * num_stages))
    
    echo "Submitting job with $nodes nodes."

    sbatch --nodes="$nodes" \
           --job-name="$job_name" \
           --output="$output_file" \
           --error="$error_file" \
           cluster_scripts/scalability_shakespeare_parsgd.job \
           "$trial" \
           "$num_epochs" \
           "$num_subdomains" \
           "$num_replicas_per_subdomain" \
           "$num_stages" \
           "$seed" \
           "$batch_size" \
           "$data_chunks_amount" \
           "$block_size" \
           "$vocab_size" \
           "$n_layer" \
           "$n_head" \
           "$n_embd" \
           "$dropout" \
           "$learning_rate" \
           "$percentage"
}

# Make sure the submit_job function is defined above this code block
for trial in "${TRIALS[@]}"; do
    seed=$((trial * 2456456))
    for num_subdomains in "${NUM_SUBDOMAINS_LIST[@]}"; do
        for num_replicas_per_subdomain in "${NUM_REPLICAS_PER_SUBDOMAIN_LIST[@]}"; do
            for num_stages in "${NUM_STAGES_LIST[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    submit_job \
                        "$trial" \
                        "$NUM_EPOCHS" \
                        "$num_subdomains" \
                        "$num_replicas_per_subdomain" \
                        "$num_stages" \
                        "$seed" \
                        "$batch_size" \
                        "$DATA_CHUNKS_AMOUNT" \
                        "$BLOCK_SIZE" \
                        "$VOCAB_SIZE" \
                        "$N_LAYER" \
                        "$N_HEAD" \
                        "$N_EMBD" \
                        "$DROPOUT" \
                        "$LEARNING_RATE" \
                        "$PERCENTAGE"
                done
            done
        done
    done
done
