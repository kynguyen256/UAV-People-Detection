

#!/bin/bash

# Navigate to the directory 
cd /app

# Create the directory for the safety set
mkdir -p $SAFETY_SET

# Download the annotation data from S3
echo "Syncing images from S3..."
mkdir -p data/annotations/$SAFETY_SET
aws s3 sync s3://didge-cv-annotation-data/safety-tracking/$SAFETY_SET data/annotations/$SAFETY_SET
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync data from S3."
  exit 1
fi

# Run your data division script
echo "Dividing data..."
python3 data/divide_data.py --safety_set $SAFETY_SET --images_annotation_split "images" # images or annotations
if [ $? -ne 0 ]; then
  echo "Error: Python data division script failed."
  exit 1
fi

# Start the continuous sync to S3 in the background
{
    while true; do
        echo "Syncing model outputs to S3... from $SAFETY_SET to s3://didge-cv-models/Co-DETR/${SAFETY_SET}_${VERSION}"
        aws s3 sync $SAFETY_SET s3://didge-cv-models/Co-DETR/${SAFETY_SET}_${VERSION} --exclude "*.pth" --include "*best_bbox_mAP*.pth" 
        if [ $? -ne 0 ]; then
            echo "Error: Failed to sync model outputs to S3."
        fi
        sleep 6000  # sleep for 100 minutes
    done
} &
PID1=$!
echo "Background sync process started with PID: $PID1"

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Get the number of available GPUs using nvidia-smi
RUNTIME_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Detected $RUNTIME_GPUS GPUs."

# Check if number of GPUs is greater than 1
if [ "$RUNTIME_GPUS" -gt 1 ]; then
    # Run the main Python training distributed script
    echo "Starting distributed training..."
    if ! python3 -m torch.distributed.launch --nproc_per_node=$RUNTIME_GPUS --master_port=29300 \
        $(dirname "$0")/train.py "$RUNTIME_CONFIG_FILE" --launcher pytorch --work-dir "$SAFETY_SET"; then
      echo "Error: Distributed Python script execution failed."
      exit 1
    fi
else
    # Run the main Python training script
    echo "Starting training..."
    if ! python3 train.py "$RUNTIME_CONFIG_FILE" --work-dir "$SAFETY_SET"; then
      echo "Error: Python script execution failed."
      exit 1
    fi
fi

# Sync the model outputs to S3 one last time
echo "Syncing model outputs to S3... from $SAFETY_SET to s3://didge-cv-models/Co-DETR/${SAFETY_SET}_${VERSION}"
aws s3 sync $SAFETY_SET s3://didge-cv-models/Co-DETR/${SAFETY_SET}_${VERSION} --exclude "*.pth" --include "*best_bbox_mAP*.pth" 

# Ensure that the sync process is terminated after the script completes:
kill $PID1
