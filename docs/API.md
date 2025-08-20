API REFERENCE
=============

CORE FUNCTIONS
--------------

Training
--------

train.py
Main training script for model training.

Arguments:
  config (str, required): Path to configuration file
  --work-dir (str): Directory for saving logs and checkpoints
  --resume-from (str): Path to checkpoint for resuming training
  --gpu-id (int): GPU device ID for training (default: 0)
  --seed (int): Random seed for reproducibility
  --auto-scale-lr: Enable automatic learning rate scaling

Example:
  python train.py configs/uav_people/co_dino_5scale_r50_1x_coco.py

Testing
-------

test.py
Model evaluation with COCO metrics.

Arguments:
  --config (str, required): Model configuration file
  --checkpoint (str, required): Model checkpoint path
  --test-img-dir (str, required): Test images directory
  --gt-json (str, required): Ground truth COCO JSON
  --output-dir (str): Output directory for results
  --score-thr (float): Score threshold for detections (default: 0.1)

Returns:
  - predictions.json: COCO format detection results
  - Metric plots: Precision, recall, F1, mIoU curves

Video Processing
----------------

video_processor.py
Process videos with object detection.

Class: VideoProcessor
  __init__(config_path, checkpoint_path, video_path, output_dir, rank=0, world_size=1)
  
Methods:
  setup(): Initialize model and environment
  extract_frames(): Extract frames from video
  run_inference(frame_paths): Run detection on frames
  create_output_video(): Compile results into video
  process(): Run complete pipeline

DATA PROCESSING
---------------

processing.py
Main data pipeline orchestrator.

Functions:
  main(): Execute complete data pipeline
  - Downloads RGB and IR datasets
  - Merges datasets
  - Creates train/valid/test splits
  - Validates annotations

merger.py
Dataset merging utilities.

Functions:
  mergeData(rgb_dataset_path, ir_dataset_path, output_path)
    Consolidates RGB and IR datasets into unified format
    
  create_splits(data_root, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_seed=42)
    Creates dataset splits
    
  validate_coco_annotations(coco_file)
    Validates COCO JSON format
    
  fixAnns(data_root='data/merged', splits=['train', 'valid', 'test'])
    Fixes annotation file paths

EVALUATION
----------

matrix.py
Confusion matrix and metrics analysis.

Functions:
  compute_iou(box1, box2)
    Calculate Intersection over Union
    
  compute_per_class_metrics(gt_data, pred_data, class_id, conf_threshold=0.5, iou_threshold=0.5)
    Calculate precision, recall, F1, mIoU per class
    
  find_optimal_threshold_per_class(gt_data, pred_data, class_id, iou_threshold=0.5)
    Find optimal confidence threshold
    
  plot_per_class_confusion_matrices(results_per_class, categories, output_dir)
    Generate confusion matrix visualizations

class_confusion_analyzer.py
Analyze confusion between IR and RGB classes.

Functions:
  analyze_class_confusion(gt_data, pred_data, conf_threshold=0.5, iou_threshold=0.5)
    Detailed class confusion analysis
    
  calculate_metrics_impact(errors_by_type, categories)
    Calculate error impact on metrics

VISUALIZATION
-------------

visualize.py
Visualization utilities.

Functions:
  randomDisplay()
    Display random sample of images
    
gendrawer.py
Generate detection visualizations.

Functions:
  visualize_images(predictions, img_dir, output_dir, num_images)
    Draw bounding boxes on images
    
heatmap.py
Create detection heatmaps.

Functions:
  create_combined_heatmap(annotation_paths)
    Generate bounding box density heatmap

MODEL CONFIGURATION
-------------------

Configuration files use Python dict format with hierarchical structure:

model = dict(
    type='CoDETR',
    backbone=dict(...),
    neck=dict(...),
    rpn_head=dict(...),
    query_head=dict(...),
    roi_head=[...],
    bbox_head=[...],
    train_cfg=[...],
    test_cfg=[...]
)

Key configuration parameters:
  num_classes: Number of detection classes (2)
  num_query: Number of object queries (900)
  num_feature_levels: Multi-scale feature levels (5)
  samples_per_gpu: Batch size per GPU
  max_iters: Maximum training iterations
  evaluation_interval: Validation frequency