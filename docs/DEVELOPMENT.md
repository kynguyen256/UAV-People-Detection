DEVELOPMENT GUIDE
=================

DEVELOPMENT SETUP
-----------------

Environment Setup:

1. Create virtual environment:
   python -m venv uav_env
   source uav_env/bin/activate  # Linux/Mac
   uav_env\Scripts\activate     # Windows

2. Install development dependencies:
   pip install -r requirements/tests.txt
   pip install -r requirements/build.txt

3. Install pre-commit hooks:
   pip install pre-commit
   pre-commit install

Project Dependencies:

Core Requirements:
- numpy==1.23.0
- torch>=1.11.0
- mmdet==2.28.1
- mmengine
- mmcv-full>=1.3.17
- pycocotools
- opencv-python>=4.1.2
- Pillow
- PyYAML>=5.3.1

Testing Requirements:
- pytest
- flake8
- isort==4.3.21
- codecov

Visualization Requirements:
- matplotlib
- seaborn>=0.11.0
- pandas

TESTING
-------

Running Tests:

Unit tests:
  pytest tests/

Code quality checks:
  flake8 src/ --max-line-length=100
  isort src/ --check-diff

Performance testing:
  python test.py --config <config> --checkpoint <checkpoint> --test-img-dir <dir> --gt-json <json>

DEBUGGING
---------

Common Issues and Solutions:

1. CUDA Out of Memory:
   - Reduce batch size in config: samples_per_gpu=1
   - Enable gradient checkpointing
   - Use mixed precision training

2. Slow Training:
   - Enable AMP: fp16 = dict(loss_scale=dict(init_scale=512))
   - Reduce number of workers: workers_per_gpu=2
   - Use SSD for dataset storage

3. Low mAP scores:
   - Check learning rate scheduling
   - Verify data augmentation pipeline
   - Ensure proper class balancing

4. Video processing errors:
   - Verify FFmpeg installation
   - Check video codec compatibility
   - Ensure sufficient disk space for frames

Debug Mode:

Enable verbose logging:
  export PYTHONUNBUFFERED=1
  export CUDA_LAUNCH_BLOCKING=1

Add debug statements in config:
  train_cfg = dict(debug=True)

BUILDING AND DEPLOYMENT
-----------------------

Model Export:

Export to ONNX:
  python tools/deployment/pytorch2onnx.py <config> <checkpoint> --output-file model.onnx

Package for distribution:
  python setup.py sdist bdist_wheel

Docker Deployment:

Build container:
  docker build -t uav-detection .

Run container:
  docker run --gpus all -v $(pwd)/data:/data uav-detection

CONFIGURATION CUSTOMIZATION
---------------------------

Creating Custom Configs:

1. Inherit from base config:
   _base_ = ['configs/_base_/models/faster_rcnn_r50_fpn.py']

2. Override specific settings:
   model = dict(
       roi_head=dict(
           bbox_head=dict(num_classes=2)
       )
   )

3. Adjust training parameters:
   optimizer = dict(lr=0.001)
   lr_config = dict(step=[8, 11])

Key Configuration Options:

Model Settings:
- num_classes: Number of detection classes
- num_query: Object queries for DETR
- num_feature_levels: Multi-scale levels

Training Settings:
- samples_per_gpu: Batch size per GPU
- workers_per_gpu: Data loading workers
- max_iters: Total training iterations
- evaluation_interval: Validation frequency

Data Pipeline:
- img_scale: Input image dimensions
- flip_ratio: Horizontal flip probability
- crop_size: Random crop dimensions

ADDING NEW FEATURES
-------------------

Adding a New Dataset:

1. Create dataset class in src/:
   class NewDataset(Dataset):
       def __init__(self, annotation_file, images_dir):
           # Initialize dataset

2. Register with MMDetection:
   @DATASETS.register_module()
   class NewDataset(CustomDataset):
       CLASSES = ('class1', 'class2')

3. Update config:
   data = dict(
       train=dict(type='NewDataset', ...)
   )

Adding a New Model:

1. Implement model class:
   @MODELS.register_module()
   class NewDetector(BaseDetector):
       def forward_train(self, img, img_metas, **kwargs):
           # Training logic

2. Create config file:
   model = dict(
       type='NewDetector',
       backbone=dict(...),
       neck=dict(...),
       head=dict(...)
   )

PERFORMANCE OPTIMIZATION
------------------------

Training Optimization:

1. Multi-GPU training:
   - Use DistributedDataParallel
   - Sync batch normalization
   - Gradient accumulation

2. Memory optimization:
   - Gradient checkpointing
   - Mixed precision training
   - Reduce batch size

3. Speed optimization:
   - Increase num_workers
   - Use persistent_workers=True
   - Enable cudnn.benchmark

Inference Optimization:

1. Model optimization:
   - TensorRT conversion
   - ONNX optimization
   - Quantization

2. Batch processing:
   - Group similar-sized images
   - Parallel preprocessing
   - Async inference

MONITORING AND LOGGING
----------------------

Wandb Integration:

Configure in training:
  log_config = dict(
      hooks=[
          dict(
              type='MMDetWandbHook',
              init_kwargs={
                  'project': 'UAV-Detection',
                  'name': 'experiment_name'
              }
          )
      ]
  )

Tensorboard Logging:

Enable in config:
  log_config = dict(
      hooks=[
          dict(type='TensorboardLoggerHook')
      ]
  )

View logs:
  tensorboard --logdir work_dirs/