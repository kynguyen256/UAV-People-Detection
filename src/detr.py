import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

def create_detr_model(device="cuda"):
    """
    Loads the DETR model for object detection with pre-trained weights and moves it to the specified device.
    Returns the model and the image processor for preprocessing images.
    
    Args:
        device (str): The device to which the model will be moved. Default is 'cuda'.
    """
    # Load the pre-trained model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Move the model to the specified device
    model = model.to(device)
    
    return model, processor
