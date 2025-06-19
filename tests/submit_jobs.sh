#!/bin/bash
set -euo pipefail

# --- Constants and Defaults --- #
SCRIPT="run_config_file.py"      # Python script
PROJECT="tr_variants_assessment" # Scheduler project name
PAPER_TR_UPDATES=(false true)    # For LSSR1-TR (inside or outside of APTS), whether to use the TR updates from the paper
TRIALS=1                         # Repetitions per configuration
USE_PMW=false                    # PMW optimizer flag
GRAD_ACC=false                   # Gradient accumulation flag
SCALING_TYPE="weak"              # "weak": scale up batch; "strong": scale down

# Parallelism sweep settings
NUM_SUBD=(1)
NUM_STAGES=(1)
NUM_REP=(1)
BATCH_SIZES=(15000)

# Configuration sweeps
OPTIMIZERS=(tr)
DATASETS=(mnist)
MODELS=(simple_ffnn)

# Second-order toggles
GLOB_SECOND_ORDERS=(true)
LOC_SECOND_ORDERS=(false true)

# Evaluation parameters: epochs, max iterations, loss
EVAL_PARAMS=(
  epochs=10
  max_iters=0
  criterion=cross_entropy
)

# Adaptive solver parameters
APTS_PARAMS=(
  batch_inc_factor=1.0
  overlap=0.0
  glob_second_order=false
)

# --- Functions to Adjust Defaults --- #

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
  local opt="$1"
  if [[ "$opt" =~ ^(apts_d|apts_p|apts_ip|lssr1_tr|tr)$ ]]; then
    APTS_PARAMS=(
      batch_inc_factor=1.5
      overlap=0.33
      max_wolfe_iters=5
      max_zoom_iters=5
      mem_length=5
    )
    if [[ "$opt" != "lssr1_tr" ]]; then
      APTS_PARAMS+=(glob_opt=lssr1_tr max_glob_iters=1 glob_second_order=true
        loc_opt=lssr1_tr max_loc_iters=3 loc_second_order=true)
      case "$opt" in
      apts_d) APTS_PARAMS+=(glob_pass=true foc=true) ;;
      apts_p) APTS_PARAMS+=(glob_pass=true foc=false) ;;
      apts_ip) APTS_PARAMS+=(loc_opt=sgd loc_second_order=false glob_pass=true) ;;
      tr) APTS_PARAMS+=(glob_second_order=false) ;;
      esac
    else
      APTS_PARAMS+=(glob_second_order=false)
    fi
  fi
}

extract_apts_details() {
  APTS_GLOB_OPT="none"
  APTS_LOC_OPT="none"
  APTS_GLOB_SO="false"
  APTS_LOC_SO="false"
  for kv in "${APTS_PARAMS[@]}"; do
    IFS="=" read -r key val <<<"$kv"
    case "$key" in
    glob_opt) APTS_GLOB_OPT=$val ;;
    loc_opt) APTS_LOC_OPT=$val ;;
    glob_second_order) APTS_GLOB_SO=$val ;;
    loc_second_order) APTS_LOC_SO=$val ;;
    esac
  done
}

submit_job() {
  local template="$1"
  local jobfile
  jobfile=$(mktemp)
  sed -e "s|\${job_name}|${job_name}|g" \
    -e "s|\${world_size}|${world_size}|g" \
    -e "s|\${script}|${SCRIPT}|g" \
    -e "s|\${num_stages}|${num_stages}|g" \
    -e "s|\${num_subd}|${num_subd}|g" \
    -e "s|\${num_rep}|${num_rep}|g" \
    -e "s|\${ntasks_per_node}|${ntasks_per_node}|g" \
    "$template" >"$jobfile"
  if ! sbatch --nodes="${nodes}" "$jobfile"; then
    echo "ERROR: sbatch failed for $job_name" >&2
  fi
  rm -f "$jobfile"
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

# --- Main Sweep Loop --- #
for optimizer in "${OPTIMIZERS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for gso in "${GLOB_SECOND_ORDERS[@]}"; do
        for lso in "${LOC_SECOND_ORDERS[@]}"; do
          for PAPER_TR_UPDATE in "${PAPER_TR_UPDATES[@]}"; do

            set_optimizer_params "$optimizer"
            set_model_params "$model"
            set_grad_acc_params
            set_hardware_params
            set_apts_lssr1_tr_params "$optimizer"
            extract_apts_details

            # Override SO flags for this sweep
            tmp=()
            for kv in "${APTS_PARAMS[@]}"; do
              key=${kv%%=*}
              if [[ $key != "glob_second_order" && $key != "loc_second_order" ]]; then
                tmp+=("$kv")
              fi
            done
            APTS_PARAMS=("${tmp[@]}" "glob_second_order=$gso" "loc_second_order=$lso")
            extract_apts_details

            for num_stages in "${NUM_STAGES[@]}"; do
              for num_subd in "${NUM_SUBD[@]}"; do
                for num_rep in "${NUM_REP[@]}"; do
                  for trial in $(seq 1 "$TRIALS"); do
                    for batch_size in "${BATCH_SIZES[@]}"; do

                      if [[ "$SCALING_TYPE" == "weak" ]]; then
                        actual_bs=$((batch_size * num_subd))
                        eff_bs=$batch_size
                      else
                        actual_bs=$batch_size
                        eff_bs=$((batch_size / num_subd))
                      fi

                      IFS="=" read -r _ EPOCH_COUNT <<<"${EVAL_PARAMS[0]}"

                      job_name="${optimizer}_${dataset}_${actual_bs}_epochs_${EPOCH_COUNT}_nst_${num_stages}_nsd_${num_subd}_nrpsd_${num_rep}"

                      if [[ "$optimizer" == *apts_* ]]; then
                        job_name+="_gopt_${APTS_GLOB_OPT}_lopt_${APTS_LOC_OPT}_gso_${APTS_GLOB_SO}_lso_${APTS_LOC_SO}"
                        if [[ "$APTS_GLOB_OPT" == "lssr1_tr" || "$APTS_LOC_OPT" == "lssr1_tr" ]]; then
                          job_name+="_ptru_${PAPER_TR_UPDATE}"
                        fi
                      elif [[ "$optimizer" == "lssr1_tr" ]]; then
                        job_name+="_gso_${APTS_GLOB_SO}_ptru_${PAPER_TR_UPDATE}"
                      elif [[ "$optimizer" == "tr" ]]; then
                        job_name+="_gso_${APTS_GLOB_SO}"
                      fi

                      job_name+="_pmw_${USE_PMW}_trial_${trial}"

                      world_size=$((num_stages * num_subd * num_rep))
                      nodes=$(calc_nodes)
                      ntasks_per_node=$((world_size / nodes))

                      config_file="./config_files/config_${job_name}.yaml"
                      if [[ -e "$config_file" ]]; then
                        echo "→ Skipping existing config: $config_file"
                        continue
                      fi
                      cp "./config_files/config_${optimizer}.yaml" "$config_file"

                      update_config batch_size "$actual_bs"
                      update_config effective_batch_size "$eff_bs"
                      update_config dataset_name "$dataset"
                      update_config model_name "$model"
                      update_config criterion "${EVAL_PARAMS[2]#*=}"
                      update_config epochs "${EVAL_PARAMS[0]#*=}"
                      update_config max_iters "${EVAL_PARAMS[1]#*=}"
                      update_config num_subdomains "$num_subd"

                      if [[ "$optimizer" == "lssr1_tr" ]] ||
                        ([[ "$optimizer" == *apts_* ]] &&
                          ([[ "$APTS_GLOB_OPT" == "lssr1_tr" ]] ||
                            [[ "$APTS_LOC_OPT" == "lssr1_tr" ]])); then
                        update_config paper_tr_update "$PAPER_TR_UPDATE"
                      fi

                      for kv in "${APTS_PARAMS[@]}"; do
                        IFS="=" read -r key val <<<"$kv"
                        update_config "$key" "$val"
                      done

                      if $USE_PMW; then
                        update_config num_stages "$num_stages"
                        update_config num_replicas_per_subdomain "$num_rep"
                      fi
                      if $GRAD_ACC; then
                        update_config gradient_accumulation true
                        update_config accumulation_steps "$ACCUM_STEPS"
                      fi

                      template=$([[ "$(pwd)" == *"/home/"* ]] && echo rosa.job || echo daintalps.job)
                      if [[ ! -f "$template" ]]; then
                        echo "ERROR: job template '$template' not found" >&2
                        exit 1
                      fi

                      export nccl_debug=WARN job_name SCRIPT use_wandb=1 \
                        num_stages num_subd num_rep world_size ntasks_per_node \
                        config_file optimizer trial PROJECT
                      submit_job "$template"

                    done
                  done
                done
              done
            done

          done
        done
      done
    done
  done
done
