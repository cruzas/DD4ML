#!/bin/bash

# Current working directory
current_dir=$(pwd)
script="run_config_file.py" # Python script to run

# --- General parameter settings ---#
optimizer="apts_ip"
dataset="mnist"
batch_sizes=(10000)
model="simple_cnn"
criterion="cross_entropy"
epochs=20
trials=1
num_subd_arr=(2) # For data-parallel executions
batch_inc_factor=1.25

# --- Optimizer-specific settings ---#
use_pmw="false"
if [[ "$optimizer" == "apts_ip" ]]; then
    use_pmw="true"
fi

if [[ "$use_pmw" == "true" ]]; then
    num_subd_arr=(1) # For data-parallel executions
    num_stages_arr=(2)
    num_rep_arr=(1)
fi

gradient_accumulation="False"
if [[ "$gradient_accumulation" == "True" ]]; then
    accumulation_steps=3
fi

# Check if optimizer is apts_d
if [[ "$optimizer" == "apts_d" ]]; then
    glob_pass="True"
    foc="False"
fi

if [[ "$current_dir" == *"home"* ]]; then
    max_ngpu_per_node=1 # 2 only for multi-gpu on USI Rosa
else
    max_ngpu_per_node=4 # each node has 4 GPUs on Daint Alps
fi

# --- Helper function for job submission ---#
submit_job() {
    local template="$1"
    local temp_job
    temp_job=$(mktemp)
    sed -e "s|\${job_name}|${job_name}|g" \
        -e "s|\${world_size}|${world_size}|g" \
        -e "s|\${script}|${script}|g" \
        -e "s|\${num_stages}|${num_stages}|g" \
        -e "s|\${num_subd}|${num_subd}|g" \
        -e "s|\${num_rep}|${num_rep}|g" \
        -e "s|\${ntasks_per_node}|${ntasks_per_node}|g" \
        "${template}" >"${temp_job}"
    sbatch --nodes=${nodes} "${temp_job}"
    rm "${temp_job}"
}

# --- Determine homogeneous node allocation ---#
calc_nodes() {
    for n in $(seq 1 $world_size); do
        if [ $((world_size % n)) -eq 0 ]; then
            tasks_per_node=$((world_size / n))
            if [ $tasks_per_node -le $max_ngpu_per_node ]; then
                echo "$n"
                return
            fi
        fi
    done
    # Fallback: one task per node
    echo "$world_size"
}

# --- Helper to update config file parameters ---#
update_config() {
    local key="$1" value="$2"
    sed -i "/${key}:/ {n; s/value: .*/value: ${value}/}" "${config_file}"
}

# --- Main job submission loop ---#
for num_stages in "${num_stages_arr[@]:-1}"; do
    for num_subd in "${num_subd_arr[@]}"; do
        for num_rep in "${num_rep_arr[@]:-1}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for trial in $(seq 1 "${trials}"); do
                    job_name="${optimizer}_${dataset}_${batch_size}_nst_${num_stages}_nsd_${num_subd}_nrpsd_${num_rep}_trial_${trial}"
                    world_size=$((num_stages * num_subd * num_rep))
                    nodes=$(calc_nodes)
                    ntasks_per_node=$((world_size / nodes))

                    config_file="./config_files/config_${job_name}.yaml"
                    cp ./config_files/config_${optimizer}.yaml "${config_file}"

                    update_config "dataset_name" "${dataset}"
                    update_config "model_name" "${model}"
                    update_config "criterion" "${criterion}"
                    update_config "epochs" "${epochs}"
                    update_config "num_subdomains" "${num_subd}"

                    if [[ "$use_pmw" == "true" ]]; then
                        update_config "num_stages" "${num_stages}"
                        update_config "num_replicas_per_subdomain" "${num_rep}"
                    fi

                    # Check if gradient accumulation is enabled
                    if [[ "$gradient_accumulation" == "True" ]]; then
                        update_config "gradient_accumulation" "${gradient_accumulation}"
                        update_config "accumulation_steps" "${accumulation_steps}"
                    fi

                    if [[ "$optimizer" == "apts_p" || "$optimizer" == "apts_ip" ]]; then
                        update_config "batch_size" "${batch_size}"
                    else
                        effective_batch_size=${batch_size}
                        actual_batch_size=$((effective_batch_size * num_subd))
                        update_config "batch_size" "${actual_batch_size}"
                        update_config "effective_batch_size" "${effective_batch_size}"
                    fi

                    if [[ "$optimizer" == "apts_d" ]]; then
                        update_config "glob_pass" "${glob_pass}"
                        update_config "foc" "${foc}"
                    fi

                    export nccl_debug=WARN job_name script use_wandb=1 num_stages num_subd num_rep world_size ntasks_per_node config_file optimizer trial
                    if [[ "$current_dir" == *"home"* ]]; then
                        submit_job rosa.job
                    else
                        submit_job daintalps.job
                    fi
                done
            done
        done
    done
done
