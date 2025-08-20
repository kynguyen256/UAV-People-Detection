UAV-People-Detection
=====================

Multi-modal aerial human detection system using CO-DETR architecture for RGB and infrared imagery analysis.

FEATURES
--------
- Dual-modal detection: Simultaneous processing of RGB and infrared (IR) imagery
- CO-DETR architecture: Collaborative DETR with DINO transformer for enhanced detection
- Two-class detection: Distinguishes between human_ir (IR imagery) and human_rgb (RGB imagery)
- Video processing: Real-time detection on video streams with overlay generation
- Distributed training: Multi-GPU support via PyTorch distributed training
- Comprehensive evaluation: COCO-format metrics with confusion matrix analysis
- Automated data pipeline: Dataset downloading, merging, and preprocessing tools

INSTALLATION
------------

Prerequisites:
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.11.0+
- MMDetection framework

Setup:

1. Clone the repository:
   git clone <repository-url>
   cd UAV-People-Detection

2. Install base dependencies:
   pip install -r requirements.txt

3. Install MMDetection dependencies:
   pip install -r requirements/build.txt
   pip install -r requirements/runtime.txt
   pip install mmdet==2.28.1
   pip install mmengine

USAGE
-----

Data Preparation:

Download and prepare the datasets:
  python processing.py

This will:
- Download RGB dataset from Roboflow
- Download IR dataset from Roboflow
- Merge datasets into unified format
- Create train/valid/test splits (70%/15%/15%)

Training:

Train the CO-DETR model:
  python train.py configs/uav_people/co_dino_5scale_r50_1x_coco.py --work-dir work_dirs/experiment_v1

Resume training from checkpoint:
  python train.py configs/uav_people/co_dino_5scale_r50_1x_coco.py --resume-from work_dirs/experiment_v1/latest.pth

Testing:

Evaluate model on test set:
  python test.py --config configs/uav_people/co_dino_5scale_r50_1x_coco.py --checkpoint work_dirs/experiment_v1/latest.pth --test-img-dir data/merged/test --gt-json data/merged/test/_annotations.coco.json --output-dir output/

Video Processing:

Process video with detection overlays:
  python video_processor.py --config configs/uav_people/co_dino_5scale_r50_1x_coco.py --checkpoint work_dirs/experiment_v1/latest.pth --video input_video.mp4 --output-dir output/

PROJECT STRUCTURE
-----------------

UAV-People-Detection/
├── configs/                  # Model and training configurations
│   ├── _base_/              # Base configurations
│   └── uav_people/          # UAV-specific configs
├── data/                    # Dataset directory (created after download)
│   ├── RGB/                 # RGB imagery dataset
│   ├── IR/                  # Infrared imagery dataset
│   └── merged/              # Combined dataset with splits
├── src/                     # Source code modules
│   ├── download_RGB.py      # RGB dataset downloader
│   ├── download_IR.py       # IR dataset downloader
│   ├── merger.py            # Dataset merging utilities
│   ├── matrix.py            # Confusion matrix analysis
│   └── visualize.py         # Visualization tools
├── mmcv_custom/             # Custom MMCV components
├── requirements/            # Dependency specifications
├── train.py                 # Main training script
├── test.py                  # Evaluation script
├── video_processor.py       # Video inference pipeline
└── processing.py            # Data pipeline orchestrator

CONTRIBUTING
------------

1. Fork the repository
2. Create a feature branch (git checkout -b feature/improvement)
3. Commit changes (git commit -am 'Add new feature')
4. Push to branch (git push origin feature/improvement)
5. Create Pull Request

LICENSE
-------

This project uses the MMDetection framework. Please refer to the original MMDetection license for framework components.