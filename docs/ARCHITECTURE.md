ARCHITECTURE OVERVIEW
=====================

SYSTEM DESIGN
-------------

The UAV-People-Detection system implements a multi-modal object detection pipeline using state-of-the-art transformer-based architectures.

CORE COMPONENTS
---------------

1. Model Architecture
---------------------

CO-DETR (Collaborative DETR):
- Base architecture with collaborative heads for improved detection
- Implements both DINO and deformable attention mechanisms
- Configured for 2-class detection: human_ir (class 0) and human_rgb (class 1)

Backbone Networks:
- ResNet-50: Default backbone with FPN (Feature Pyramid Network)
- Swin Transformer: Optional large model for enhanced performance
- Both support multi-scale feature extraction (5 scales)

Detection Heads:
- Query Head: CoDINOHead with 900 queries, mixed selection strategy
- RPN Head: Region Proposal Network for auxiliary supervision
- ROI Head: CoStandardRoIHead for refined bounding box predictions
- ATSS Head: Anchor-based head for additional supervision

2. Data Pipeline
----------------

Input Processing:
- Dual-stream processing for RGB and IR imagery
- COCO format annotations with custom category mapping
- Automated augmentation including resize, flip, and normalization

Dataset Structure:
  data/merged/
  ├── train/           # 70% of data
  ├── valid/           # 15% of data
  └── test/            # 15% of data

3. Training Configuration
-------------------------

Optimization:
- AdamW optimizer with learning rate 2e-4
- Gradient clipping (max_norm=0.1)
- Automatic mixed precision support
- Checkpoint saving at regular intervals

Loss Functions:
- Classification: QualityFocalLoss for query head
- Regression: L1Loss and GIoULoss combination
- Multi-head supervision with weighted losses

PROCESSING FLOW
---------------

1. Data Loading: Images loaded from merged dataset
2. Preprocessing: Normalization, resizing, padding
3. Feature Extraction: Backbone network processes images
4. Neck Processing: FPN or ChannelMapper creates multi-scale features
5. Detection: Multiple detection heads generate predictions
6. Post-processing: NMS and score thresholding
7. Output: Bounding boxes with class labels and confidence scores

DISTRIBUTED TRAINING
--------------------

The system supports multi-GPU training through PyTorch's DistributedDataParallel:
- Automatic process group initialization
- Gradient synchronization across GPUs
- Configurable batch sizes per GPU

TRANSFORMER ARCHITECTURE
------------------------

Encoder:
- 6 layers with deformable attention
- Multi-scale feature processing
- Checkpoint gradient for memory efficiency

Decoder:
- 6 layers with self and cross attention
- Iterative box refinement
- Dynamic query selection

MULTI-MODAL FUSION
------------------

The system processes RGB and IR imagery through:
- Separate feature extraction paths
- Unified detection framework
- Class-specific output heads

Performance optimizations include:
- Mixed precision training
- Gradient accumulation
- Memory-efficient attention mechanisms