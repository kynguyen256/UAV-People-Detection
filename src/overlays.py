import json
import cv2
from pathlib import Path

def load_annotations(file_name):
    """
    Load annotations from a COCO JSON file.
    
    Parameters:
    - file_name: Path to the annotation file.
    
    Returns:
    - The loaded annotations as a dictionary.
    """
    with open(file_name, 'r') as file:
        return json.load(file)

def overlay_bounding_boxes(image_dir, bbox_data, image_id_to_file, output_dir, color=(0, 255, 0), thickness=2):
    """
    Overlay bounding boxes on images and save the result to the output directory.
    
    Parameters:
    - image_dir: Path to the directory containing images.
    - bbox_data: Dictionary with image IDs as keys and lists of bounding boxes as values.
    - image_id_to_file: Dictionary mapping image IDs to file names.
    - output_dir: Path to the directory where the output images will be saved.
    - color: Bounding box color as a BGR tuple (default: green).
    - thickness: Bounding box line thickness (default: 2).
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(image_dir)

    for image_id, bboxes in bbox_data.items():
        # Get the image file name
        if image_id not in image_id_to_file:
            print(f"Image ID {image_id} not found in mapping.")
            continue

        file_name = image_id_to_file[image_id]
        image_path = image_dir / file_name

        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        # Draw bounding boxes
        for bbox in bboxes:
            x, y, width, height = map(int, bbox)
            top_left = (x, y)
            bottom_right = (x + width, y + height)
            cv2.rectangle(image, top_left, bottom_right, color, thickness)

        # Save the output image
        output_path = output_dir / file_name
        cv2.imwrite(str(output_path), image)

# Example usage

def createOverlays():
    # Define file paths
    file_paths = {
        'test': 'data/test/_annotations.coco.json',
        'train': 'data/train/_annotations.coco.json',
        'valid': 'data/valid/_annotations.coco.json'
    }

    # Load the annotations
    annotations = {key: load_annotations(path) for key, path in file_paths.items()}

    # Extract images and annotations
    images = annotations['valid']['images']
    annotations_data = annotations['valid']['annotations']

    # Create a mapping from image IDs to image file names
    image_id_to_file = {img['id']: img['file_name'] for img in images}

    # Extract bounding boxes and associated image IDs
    bbox_data = {}
    for ann in annotations_data:
        image_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        if image_id not in bbox_data:
            bbox_data[image_id] = []
        bbox_data[image_id].append(bbox)

    # Specify image and output directories
    image_dir = '/content/UAV-People-Detection/data/valid'
    output_dir = '/content/UAV-People-Detection/data/annotated'

    # Overlay bounding boxes and save images
    overlay_bounding_boxes(image_dir, bbox_data, image_id_to_file, output_dir, color=(0, 255, 0), thickness=2)

