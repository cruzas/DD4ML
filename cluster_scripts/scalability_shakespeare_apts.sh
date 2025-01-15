#!/bin/bash

RUNPATH=/scratch/snx3000/scruzale/DD4ML/
cd $RUNPATH || exit

# Fixed parameters
NUM_EPOCHS=15
DATA_CHUNKS_AMOUNT=1
BLOCK_SIZE=256
VOCAB_SIZE=0
N_LAYER=6
N_HEAD=6
N_EMBD=384
DROPOUT=0.0
LEARNING_RATE=0.01
PERCENTAGE=25.0
NUM_WORKERS=(0)
SDI=5 # number of subdomain iterations

# Arrays of parameters that change
TRIALS=(0)
NUM_SUBDOMAINS_LIST=(2 4 8)
NUM_REPLICAS_PER_SUBDOMAIN_LIST=(1)
NUM_STAGES_LIST=(2) # total layers = 3+n_layer*(3+n_head) 
BATCH_SIZES=(128)

LOG_DIR="./results/log_files"

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
    local num_workers=${17}
    local sdi=${18}

    echo "Submitting job with trial=$trial, num_epochs=$num_epochs, sdi=$sdi, num_subdomains=$num_subdomains, num_replicas_per_subdomain=$num_replicas_per_subdomain, num_stages=$num_stages, seed=$seed, batch_size=$batch_size, data_chunks_amount=$data_chunks_amount, block_size=$block_size, vocab_size=$vocab_size, n_layer=$n_layer, n_head=$n_head, n_embd=$n_embd, dropout=$dropout, learning_rate=$learning_rate, percentage=$percentage, num_workers=$num_workers"

    # Make a string replacing lr "." with "_"
    lr_str=$(echo "$learning_rate" | tr . _)
    perc_str=$(echo "$percentage" | tr . _)

    job_name="ts_t_${trial}_nw_${num_workers}_bls_${BLOCK_SIZE}_nl_${N_LAYER}_nh_${N_HEAD}_ne_${N_EMBD}_ns_${num_subdomains}_nr_${num_replicas_per_subdomain}_st_${num_stages}_bs_${batch_size}_dc_${DATA_CHUNKS_AMOUNT}_lr_${lr_str}_perc_${perc_str}_sdi_${sdi}_epochs_${NUM_EPOCHS}_apts"
    error_file="$LOG_DIR/${job_name}.err"
    output_file="$LOG_DIR/${job_name}.out"

    # Calculate number of nodes
    nodes=$((num_subdomains * num_replicas_per_subdomain * num_stages))
    
    echo "Submitting job with $nodes nodes."

    sbatch --nodes="$nodes" \
           --job-name="$job_name" \
           --output="$output_file" \
           --error="$error_file" \
           cluster_scripts/scalability_shakespeare_apts.job \
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
           "$percentage" \
           "$num_workers" \
           "$sdi"
}

# Make sure the submit_job function is defined above this code block
for trial in "${TRIALS[@]}"; do
    seed=$((trial * 2456456))
    for num_subdomains in "${NUM_SUBDOMAINS_LIST[@]}"; do
        for num_replicas_per_subdomain in "${NUM_REPLICAS_PER_SUBDOMAIN_LIST[@]}"; do
            for num_stages in "${NUM_STAGES_LIST[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    for num_workers in "${NUM_WORKERS[@]}"; do
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
                        "$PERCENTAGE" \
                        "$num_workers" \
                        "$SDI"
                    done
                done
            done
        done
    done
done
