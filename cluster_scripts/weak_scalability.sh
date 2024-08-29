#!/bin/bash

# Weak Scaling Test Script

# Fixed parameters
MODELS=("NN")
DATASETS=("MNIST")
EPOCHS=10
BASE_BATCH_SIZE=32 # Scaled by NUM_GPUS later
NUM_GPUS=(1 2 4 8)

# Output log directory
LOG_DIR="../scaling_logs/weak_scaling_logs"

# Check if LOG_DIR exists. If not, create it.
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi

# Number of GPUs to test
for NUM_GPUS in "${NUM_GPUS[@]}"
do
    echo "Running with $NUM_GPUS GPUs"

    # Adjust the dataset path or batch size proportionally
    DATASET_PATH="${BASE_DATASET_PATH}_scaled_${NUM_GPUS}"
    BATCH_SIZE=$(($BASE_BATCH_SIZE * $NUM_GPUS))

    # Launch the training with the scaled dataset/batch size
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        $MODEL \
        --dataset $DATASET_PATH \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir $LOG_DIR/weak_scaling_${NUM_GPUS}_gpus.log

    echo "Completed run with $NUM_GPUS GPUs"
done
