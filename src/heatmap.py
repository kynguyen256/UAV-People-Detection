import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO

def create_combined_heatmap(annotation_paths):
    # Define the heatmap resolution (e.g., 1024x1024 pixels)
    heatmap_resolution = (1024, 1024)
    heatmap = np.zeros(heatmap_resolution, dtype=np.float32)
    
    # Process each annotation file
    for annotation_file in annotation_paths:
        # Load the COCO annotations
        coco = COCO(annotation_file)
        
        # Iterate over each annotation
        for ann_id in coco.getAnnIds():
            # Get the annotation
            annotation = coco.loadAnns(ann_id)[0]
            
            # Extract bounding box coordinates (x, y, width, height)
            x, y, width, height = annotation['bbox']
            
            # Convert bounding box to fit heatmap resolution
            x = int(x * heatmap_resolution[1] / coco.imgs[annotation['image_id']]['width'])
            y = int(y * heatmap_resolution[0] / coco.imgs[annotation['image_id']]['height'])
            width = int(width * heatmap_resolution[1] / coco.imgs[annotation['image_id']]['width'])
            height = int(height * heatmap_resolution[0] / coco.imgs[annotation['image_id']]['height'])
            
            # Increment heatmap values in the bounding box area
            heatmap[y:y + height, x:x + width] += 1
    
    # Normalize heatmap to the range [0, 1]
    heatmap /= np.max(heatmap)
    
    # Display the heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Combined Bounding Box Heatmap")
    plt.show()

    # Convert heatmap to 8-bit (0-255) scale for saving as an image
    heatmap_uint8 = np.uint8(255 * heatmap)
    cv2.imwrite('combined_bounding_box_heatmap.png', heatmap_uint8)
    print("Combined heatmap saved to: combined_bounding_box_heatmap.png")

def store_COCO():
    splits = ['train', 'valid', 'test']
    paths = []
    for split in splits:
        input_annotation_path = f'data/{split}/_annotations.coco.json'
        paths.append(input_annotation_path)
    return paths

def heatmaps():
    annotations = store_COCO()
    create_combined_heatmap(annotations)  # Pass all annotation paths to the heatmap function

