import json
import os
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

# Run the process_dataset function
process_dataset()
