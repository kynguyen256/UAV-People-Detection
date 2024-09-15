import os
import json
import cv2
import glob
from pathlib import Path


def load_annotations(file_name):
    """
    Load annotations from a COCO JSON file.
    """
    with open(file_name, 'r') as file:
        return json.load(file)

def save_annotations(annotations, file_name):
    """
    Save annotations to a COCO JSON file.
    """
    with open(file_name, 'w') as file:
        json.dump(annotations, file)

def filter_humans_annotations(annotations):
    """
    Filter annotations to keep only the 'humans' class.
    """
    # Find the 'humans' category_id
    categories = annotations['categories']
    humans_category = None
    for category in categories:
        if category['name'] == 'humans':
            humans_category = category
            break

    if humans_category is None:
        print("Error: 'humans' class not found in categories.")
        return None

    humans_category_id = humans_category['id']

    # Filter annotations to keep only those with 'humans' category_id
    filtered_annotations = [ann for ann in annotations['annotations'] if ann['category_id'] == humans_category_id]

    # Update the 'categories' section to keep only 'humans' category
    annotations['categories'] = [humans_category]

    # Update 'annotations' section with filtered annotations
    annotations['annotations'] = filtered_annotations

    # Re-map category_id to 1 for consistency
    new_category_id = 1
    annotations['categories'][0]['id'] = new_category_id
    for ann in annotations['annotations']:
        ann['category_id'] = new_category_id

    return annotations

def process_dataset():
    """
    Processes the dataset to keep only 'humans' class and saves the updated annotations.
    """
    # Define dataset splits
    splits = ['train', 'valid', 'test']
    for split in splits:
        input_annotation_path = f'data/{split}/_annotations.coco.json'
        output_annotation_path = f'data/{split}/{split}_annotations.coco.json'

        # Load annotations
        annotations = load_annotations(input_annotation_path)

        # Print class names before filtering
        categories = annotations['categories']
        class_names = [category['name'] for category in categories]
        print(f"Class names in {split} set before filtering:")
        print("Class names:", class_names)

        # Filter annotations to keep only 'humans' class
        filtered_annotations = filter_humans_annotations(annotations)
        if filtered_annotations is None:
            print(f"Skipping {split} set due to missing 'humans' class.")
            continue

        # Print class names after filtering
        categories = filtered_annotations['categories']
        class_names = [category['name'] for category in categories]
        print(f"Class names in {split} set after filtering:")
        print("Class names:", class_names)

        # Save updated annotations with new filenames
        save_annotations(filtered_annotations, output_annotation_path)
        print(f"Saved filtered annotations to {output_annotation_path}\n")

def overlay_bounding_boxes(image_dir, bbox_data, image_id_to_file, output_dir, category_id_to_name, color=(0, 255, 0), thickness=2):
    """
    Overlay bounding boxes on images and save the result to the output directory.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(image_dir)

    for image_id, annotations in bbox_data.items():
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

        # Draw bounding boxes and class names
        for ann in annotations:
            bbox = ann['bbox']
            x, y, width, height = map(int, bbox)
            top_left = (x, y)
            bottom_right = (x + width, y + height)
            cv2.rectangle(image, top_left, bottom_right, color, thickness)

            # Get the class name
            category_id = ann['category_id']
            class_name = category_id_to_name.get(category_id, "Unknown")

            # Put the class name text above the bounding box
            cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Save the output image
        output_path = output_dir / file_name
        cv2.imwrite(str(output_path), image)

def createOverlays():
    """
    Creates overlays on the images using the filtered annotations.
    """
    # Define file paths to the new annotations
    file_paths = {
        'test': 'data/test/test_annotations.coco.json',
        'train': 'data/train/train_annotations.coco.json',
        'valid': 'data/valid/valid_annotations.coco.json'
    }

    # Process each dataset split
    for split in ['train', 'valid', 'test']:
        # Load the annotations
        annotations = load_annotations(file_paths[split])

        # Extract images and annotations
        images = annotations['images']
        annotations_data = annotations['annotations']

        # Create a mapping from image IDs to image file names
        image_id_to_file = {img['id']: img['file_name'] for img in images}

        # Create a mapping from category IDs to names
        category_id_to_name = {cat['id']: cat['name'] for cat in annotations['categories']}

        # Extract bounding boxes and associated image IDs
        bbox_data = {}
        for ann in annotations_data:
            image_id = ann['image_id']
            if image_id not in bbox_data:
                bbox_data[image_id] = []
            bbox_data[image_id].append(ann)

        # Specify image and output directories
        image_dir = f'data/{split}'
        output_dir = f'data/{split}_annotated'

        # Overlay bounding boxes and save images
        overlay_bounding_boxes(image_dir, bbox_data, image_id_to_file, output_dir, category_id_to_name, color=(0, 255, 0), thickness=2)

        print(f"Overlays created for {split} set and saved to {output_dir}")

if __name__ == "__main__":

    # Step 2: Process the dataset to keep only the 'humans' class
    process_dataset()

    # Step 4: Create overlays
    createOverlays()
