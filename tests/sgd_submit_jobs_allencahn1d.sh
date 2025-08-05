#!/bin/bash
set -euo pipefail

DEBUGGING=true # Set to true for debugging mode

# --- Constants and Defaults --- #
SCRIPT="run_config_file.py"

if $DEBUGGING; then
    PROJECT="debugging"
    TRIALS=1
    partition="debug"
    time="00:10:00"
    SCALING_TYPE="strong" # weak or strong
    if [[ "$SCALING_TYPE" == "weak" ]]; then
        BATCH_SIZES=(128 256 512)
    else
        # For strong scaling, we use larger batch sizes
        BATCH_SIZES=(512)
  fi
    NUM_SUBD=(1)
    NUM_STAGES=(2)
    NUM_REP=(1)
else
    PROJECT="thesis_results" # thesis_results
    TRIALS=3
    partition="normal"
    time="00:20:00"

    SCALING_TYPE="weak"
    if [[ "$SCALING_TYPE" == "weak" ]]; then
        BATCH_SIZES=(64)
    else
        # For strong scaling, we use larger batch sizes
        BATCH_SIZES=(64 128 256)
    fi
    NUM_SUBD=(2 4 8)
    NUM_STAGES=(1)
    NUM_REP=(1)
fi

USE_PMW=false
GRAD_ACC=false

# --- Sweep settings: SGD only + one LR --- #
OPTIMIZERS=(sgd)
LEARNING_RATES=(0.01)
overlap=0.0
batch_inc_factor=1.0

DATASETS=(allencahn1d)
MODELS=(pinn_ffnn)

# (Remove all APTS / TR / dogleg loops – they’re skipped since optimizer=sgd)

EVAL_PARAMS=(epochs=10 max_iters=0 criterion=pinn_allencahn)

set_optimizer_params() {
    local opt="$1"
    if [[ "$opt" == "apts_ip" ]]; then
        USE_PMW=true
        NUM_SUBD=(1)
        NUM_STAGES=(2)
        NUM_REP=(1)
    fi
}

set_model_params() {
    local mdl="$1"
    if [[ "$mdl" == "nanogpt" ]]; then
        EVAL_PARAMS=(epochs=0 max_iters=2000 criterion=cross_entropy_transformers)
        BATCH_SIZES=(128)
    fi
}

set_grad_acc_params() {
    if $GRAD_ACC; then
        ACCUM_STEPS=1
    fi
}

set_hardware_params() {
    if [[ "$(pwd)" == *"/home/"* ]]; then
        MAX_GPUS=1
    else
        MAX_GPUS=4
    fi
}

submit_job() {
    local template="$1" jobfile
    jobfile=$(mktemp)
    sed -e "s|\${job_name}|${job_name}|g" \
        -e "s|\${world_size}|${world_size}|g" \
        -e "s|\${script}|${SCRIPT}|g" \
        -e "s|\${num_stages}|${num_stages}|g" \
        -e "s|\${num_subd}|${num_subd}|g" \
        -e "s|\${num_rep}|${num_rep}|g" \
        -e "s|\${ntasks_per_node}|${ntasks_per_node}|g" \
        -e "s|\${partition}|${partition}|g" \
        -e "s|\${time}|${time}|g" \
        "$template" >"$jobfile"
    if ! sbatch --nodes="${nodes}" "$jobfile"; then
        echo "ERROR: sbatch failed for $job_name" >&2
    fi
    rm -f "$jobfile"
}

calc_nodes() {
    for n in $(seq 1 "$world_size"); do
        local tpn=$((world_size / n))
        if ((world_size % n == 0 && tpn <= MAX_GPUS)); then
            echo $n
            return
        fi
    done
    echo "$world_size"
}

update_config() {
    sed -i "/$1:/ {n; s/value: .*/value: $2/}" "$config_file"
}

# --- Main Sweep Loop --- #
for optimizer in "${OPTIMIZERS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for num_stages in "${NUM_STAGES[@]}"; do
                for num_subd in "${NUM_SUBD[@]}"; do
                    for num_rep in "${NUM_REP[@]}"; do
                        for trial in $(seq 1 "$TRIALS"); do
                            for batch_size in "${BATCH_SIZES[@]}"; do

                                # Compute actual vs. effective batch size
                                if [[ "$SCALING_TYPE" == "weak" ]]; then
                                    actual_bs=$((batch_size * num_subd))
                                    eff_bs=$batch_size
                                else
                                    actual_bs=$batch_size
                                    eff_bs=$((batch_size / num_subd))
                                fi

                                set_optimizer_params "$optimizer"
                                set_model_params "$model"
                                set_grad_acc_params
                                set_hardware_params

                                # Iterate over learning rates
                                for lr in "${LEARNING_RATES[@]}"; do

                                    IFS="=" read -r _ EPOCH_COUNT <<<"${EVAL_PARAMS[0]}"

                                    job_name="${optimizer}_${dataset}_${model}_${actual_bs}_epochs_${EPOCH_COUNT}_nsd_${num_subd}_lr_${lr}_overlap_${overlap}_bif_${batch_inc_factor}_trial_${trial}"

                                    world_size=$((num_stages * num_subd * num_rep))
                                    nodes=$(calc_nodes)
                                    ntasks_per_node=$((world_size / nodes))

                                    config_file="./config_files/config_${job_name}.yaml"
                                    [[ -e "$config_file" ]] && {
                                        echo "-> Skipping existing: $config_file"
                                        continue
                                    }
                                    cp "./config_files/config_${optimizer}.yaml" "$config_file"

                                    update_config batch_size "$actual_bs"
                                    update_config effective_batch_size "$eff_bs"
                                    update_config dataset_name "$dataset"
                                    update_config model_name "$model"
                                    update_config criterion "${EVAL_PARAMS[2]#*=}"
                                    update_config epochs "${EVAL_PARAMS[0]#*=}"
                                    update_config max_iters "${EVAL_PARAMS[1]#*=}"
                                    update_config num_subdomains "$num_subd"

                                    # NEW: set learning rate
                                    update_config learning_rate "$lr"

                                    update_config num_stages "$num_stages"
                                    update_config num_replicas_per_subdomain "$num_rep"
                                    update_config overlap "$overlap"
                                    update_config batch_inc_factor "$batch_inc_factor"

                                    template=$([[ "$(pwd)" == *"/home/"* ]] && echo rosa.job || echo daintalps.job)
                                    [[ ! -f "$template" ]] && {
                                        echo "ERROR: template '$template' not found" >&2
                                        exit 1
                                    }

                                    export nccl_debug=WARN job_name SCRIPT use_wandb=1 \
                                        num_stages num_subd num_rep world_size ntasks_per_node \
                                        config_file optimizer trial PROJECT
                                    submit_job "$template"
                                done # lr
                            done     # batch_size
                        done         # trial
                    done             # num_rep
                done                 # num_subd
            done                     # num_stages
        done                         # model
    done                             # dataset
done                                 # optimizer
