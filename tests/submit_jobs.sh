#!/bin/bash

# --- Constants & Defaults --- #
SCRIPT="run_config_file.py"
TRIALS=1
USE_PMW=false
GRAD_ACC=false

# Default arrays
NUM_SUBD=(2)
NUM_STAGES=(1)
NUM_REP=(1)
BATCH_SIZES=(15000)

# Default general parameters
OPT_PARAMS=(
  optimizer=lssr1_tr
  dataset=mnist
  model=simple_ffnn
)

EVAL_PARAMS=(
  epochs=50
  max_iters=0
  criterion=cross_entropy
)

APTS_PARAMS=(
  batch_inc_factor=1.0
  overlap=0.0
  glob_second_order=false
)

# --- Functions to override defaults --- #

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
  if [[ "${OPT_PARAMS[2]#*=}" == "nanogpt" ]]; then
    EVAL_PARAMS=(epochs=0 max_iters=2000000 criterion=cross_entropy_transformers)
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

set_apts_lssr1_tr_params() {
  local opt="${OPT_PARAMS[0]#*=}"
  if [[ "$opt" =~ ^(apts_d|apts_p|apts_ip|lssr1_tr)$ ]]; then
    APTS_PARAMS=(batch_inc_factor=1.5 overlap=0.33 max_wolfe_iters=5 max_zoom_iters=5 mem_length=5)
    if [[ "$opt" != "lssr1_tr" ]]; then
      APTS_PARAMS+=(glob_opt=tr max_glob_iters=1 glob_second_order=false loc_opt=tr max_loc_iters=1 loc_second_order=false)
      if [[ "$opt" == "apts_d" ]]; then
        APTS_PARAMS+=(glob_pass=true foc=true)
      elif [[ "$opt" == "apts_p" ]]; then
        APTS_PARAMS+=(glob_pass=true foc=false)
      elif [[ "$opt" == "apts_ip" ]]; then
        APTS_PARAMS+=(loc_opt=sgd loc_second_order=false glob_pass=true)
      fi
    else
      APTS_PARAMS+=(glob_second_order=false)
    fi
  fi
}

# --- Helpers --- #

submit_job() {
  local template=$1 jobfile
  jobfile=$(mktemp)
  sed -e "s|\${job_name}|${job_name}|g" \
    -e "s|\${world_size}|${world_size}|g" \
    -e "s|\${script}|${SCRIPT}|g" \
    -e "s|\${num_stages}|${num_stages}|g" \
    -e "s|\${num_subd}|${num_subd}|g" \
    -e "s|\${num_rep}|${num_rep}|g" \
    -e "s|\${ntasks_per_node}|${ntasks_per_node}|g" \
    "$template" >"$jobfile"
  sbatch --nodes="${nodes}" "$jobfile"
  rm "$jobfile"
}

calc_nodes() {
  for n in $(seq 1 $world_size); do
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

# --- Initialise all params --- #
eval "${OPT_PARAMS[@]}"
set_optimizer_params "$optimizer"
set_model_params
set_grad_acc_params
set_hardware_params
set_apts_lssr1_tr_params

# --- Main loop --- #
for num_stages in "${NUM_STAGES[@]}"; do
  for num_subd in "${NUM_SUBD[@]}"; do
    for num_rep in "${NUM_REP[@]}"; do
      for batch_size in "${BATCH_SIZES[@]}"; do
        for trial in $(seq 1 "$TRIALS"); do
          job_name="${optimizer}_${dataset}_${batch_size}_nst_${num_stages}_nsd_${num_subd}_nrpsd_${num_rep}_trial_${trial}"
          world_size=$((num_stages * num_subd * num_rep))
          nodes=$(calc_nodes)
          ntasks_per_node=$((world_size / nodes))
          config_file="./config_files/config_${job_name}.yaml"
          cp "./config_files/config_${optimizer}.yaml" "$config_file"

          # Core updates
          update_config dataset_name "$dataset"
          update_config model_name "$model"
          update_config criterion "${EVAL_PARAMS[2]#*=}"
          update_config epochs "${EVAL_PARAMS[0]#*=}"
          update_config max_iters "${EVAL_PARAMS[1]#*=}"
          update_config num_subdomains "$num_subd"

          # APTS/LSSR1_TR updates
          for kv in "${APTS_PARAMS[@]}"; do
            IFS="=" read -r key val <<<"$kv"
            update_config "$key" "$val"
          done

          # PMW and gradient accumulation
          if $USE_PMW; then
            update_config num_stages "$num_stages"
            update_config num_replicas_per_subdomain "$num_rep"
          fi
          if $GRAD_ACC; then
            update_config gradient_accumulation true
            update_config accumulation_steps "$ACCUM_STEPS"
          fi

          # Batch‐size logic
          if [[ "$optimizer" =~ ^(apts_p|apts_ip)$ ]]; then
            update_config batch_size "$batch_size"
          else
            eff_bs=$batch_size
            update_config effective_batch_size "$eff_bs"
            update_config batch_size $((eff_bs * num_subd))
          fi

          # Submit
          export nccl_debug=WARN job_name SCRIPT use_wandb=1 \
            num_stages num_subd num_rep world_size ntasks_per_node config_file optimizer trial
          template=$([[ "$(pwd)" == *"/home/"* ]] && echo rosa.job || echo daintalps.job)
          submit_job "$template"
        done
      done
    done
  done
done
