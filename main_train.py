# main_train.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

from src.model import create_model

def load_annotations(annotation_file, images_dir):
    """
    Load annotations and prepare data.
    
    Parameters:
    - annotation_file: Path to the COCO annotation JSON file.
    - images_dir: Directory containing images.
    
    Returns:
    - images: List of image file paths.
    - bboxes: List of bounding boxes (normalized).
    - labels: List of labels (1 for 'humans').
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Mapping from image IDs to file names
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Collect images, bounding boxes, and labels
    images = []
    bboxes = []
    labels = []
    
    for ann in data['annotations']:
        image_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        
        # Get the file name
        file_name = id_to_filename[image_id]
        image_path = os.path.join(images_dir, file_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            continue
        
        # Load image to get dimensions
        img = load_img(image_path)
        width, height = img.size
        
        # Normalize bounding box coordinates
        x_min = bbox[0] / width
        y_min = bbox[1] / height
        x_max = (bbox[0] + bbox[2]) / width
        y_max = (bbox[1] + bbox[3]) / height
        bbox_normalized = [x_min, y_min, x_max, y_max]
        
        images.append(image_path)
        bboxes.append(bbox_normalized)
        labels.append(1)  # Since we have only one class
    
    return images, np.array(bboxes), np.array(labels)

def data_generator(images, bboxes, labels, batch_size=32, input_size=(224, 224)):
    """
    A generator that yields batches of images and labels.
    """
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = []
            batch_bboxes = []
            batch_labels = []
            
            batch_files = images[i:i+batch_size]
            batch_bboxes_values = bboxes[i:i+batch_size]
            batch_labels_values = labels[i:i+batch_size]
            
            for img_path, bbox, label in zip(batch_files, batch_bboxes_values, batch_labels_values):
                # Load and preprocess image
                img = load_img(img_path, target_size=input_size)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                
                batch_images.append(img_array)
                batch_bboxes.append(bbox)
                batch_labels.append(label)
            
            yield np.array(batch_images), {'bbox_output': np.array(batch_bboxes), 'class_output': np.array(batch_labels)}

def main():
    # Paths
    train_images_dir = 'data/train'
    train_annotation_file = 'data/train/train_annotations.coco.json'
    valid_images_dir = 'data/valid'
    valid_annotation_file = 'data/valid/valid_annotations.coco.json'
    
    # Load data
    print("Loading training data...")
    train_images, train_bboxes, train_labels = load_annotations(train_annotation_file, train_images_dir)
    
    print("Loading validation data...")
    valid_images, valid_bboxes, valid_labels = load_annotations(valid_annotation_file, valid_images_dir)
    
    # Create model
    model = create_model()
    model.summary()
    
    # Compile model
    loss = {
        'bbox_output': 'mse',
        'class_output': 'binary_crossentropy'
    }
    metrics = {
        'bbox_output': 'mae',
        'class_output': 'accuracy'
    }
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics
    )
    
    # Training parameters
    batch_size = 16
    epochs = 10
    steps_per_epoch = len(train_images) // batch_size
    validation_steps = len(valid_images) // batch_size
    
    # Data generators
    train_generator = data_generator(train_images, train_bboxes, train_labels, batch_size)
    valid_generator = data_generator(valid_images, valid_bboxes, valid_labels, batch_size)
    
    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=validation_steps
    )
    
    # Save the model
    model.save('human_detector_resnet50.h5')
    print("Model saved as human_detector_resnet50.h5")

if __name__ == '__main__':
    main()
