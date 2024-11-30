#!/bin/bash

# Navigate to the directory
cd /app

# Define the working directory
UAV_WORK_DIR="uav_work_dir"

# Create the directory for the UAV work
mkdir -p $UAV_WORK_DIR

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Always train on one GPU with the specific config
echo "Starting training..."
if ! python3 train.py configs/co_dino_vit/co_dino_5scale_vit_large_coco.py --work-dir "$UAV_WORK_DIR"; then
  echo "Error: Python script execution failed."
  kill $PID1
  exit 1
fi


# Ensure that the sync process is terminated after the script completes:
kill $PID1
