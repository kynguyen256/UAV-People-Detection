# src/detr.py

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

def create_detr_model():
    """
    Loads the DETR model for object detection with pre-trained weights.
    Returns the model and the image processor for preprocessing images.
    """
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    return model, processor
