#!/bin/bash

# Define the working directory
UAV_WORK_DIR="uav_work_dir"

# Create the directory for the UAV work
mkdir -p $UAV_WORK_DIR

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Always train on one GPU with the specific config
echo "Starting training..."
if ! training/bin/python train.py configs/uav_people/co_dino_5scale_r50_1x_coco.py --work-dir "$UAV_WORK_DIR"; then
  echo "Error: Python script execution failed."
  kill $PID1
  exit 1
fi

# Ensure that the sync process is terminated after the script completes:
kill $PID1
