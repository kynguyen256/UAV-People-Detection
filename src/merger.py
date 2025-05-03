import os
import json
import shutil
from pathlib import Path
import random

def mergeData(rgb_dataset_path, ir_dataset_path, output_path):
    """
    Consolidate RGB and IR datasets, each with subfolders (RGB: train, test, val; IR: train, val),
    into a single folder with unified COCO annotations. Assign category ID 0 to 'human_ir' (IR people)
    and ID 1 to 'human_rgb' (RGB people). Assign new sequential image IDs starting from 0.
    
    Args:
        rgb_dataset_path (str): Path to RGB dataset folder containing train, test, val subfolders
        ir_dataset_path (str): Path to IR dataset folder containing train, val subfolders
        output_path (str): Path to output consolidated dataset
    """
    output_path = Path(output_path)
    output_images = output_path / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    coco_output = {
        "info": {"description": "Consolidated RGB and IR Dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "human_ir", "supercategory": "person"},
            {"id": 1, "name": "human_rgb", "supercategory": "person"}
        ]
    }

    # Track new image and annotation IDs
    new_image_id = 0
    new_annotation_id = 0
    filename_to_new_id = {}  # Map output filename to new image ID

    # Define subfolders for each dataset
    rgb_subfolders = ["train", "test", "valid"]
    ir_subfolders = ["train", "valid"]

    # Process RGB dataset
    for subfolder in rgb_subfolders:
        dataset_path = Path(rgb_dataset_path) / subfolder
        annotation_file = dataset_path / "_annotations.coco.json"

        if not annotation_file.exists():
            print(f"Annotation file not found in {dataset_path}")
            continue

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        category_map = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        orig_label = "humans"  # Adjust if your RGB category name differs
        if orig_label not in category_map:
            print(f"Category '{orig_label}' not found in RGB {subfolder} dataset")
            continue
        person_cat_id = category_map[orig_label]

        # Process images
        for img in coco_data["images"]:
            src_img_path = dataset_path / img["file_name"]
            dst_img_name = f"RGB_{subfolder}_{img['file_name']}"
            dst_img_path = output_images / dst_img_name

            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Image {src_img_path} not found, skipping")
                continue

            # Assign new image ID
            filename_to_new_id[dst_img_name] = new_image_id
            img_copy = img.copy()
            img_copy["id"] = new_image_id
            img_copy["file_name"] = str(dst_img_path.relative_to(output_path))
            coco_output["images"].append(img_copy)
            new_image_id += 1

        # Process annotations
        for ann in coco_data["annotations"]:
            if ann["category_id"] == person_cat_id:
                old_img = next(i for i in coco_data["images"] if i["id"] == ann["image_id"])
                old_filename = f"RGB_{subfolder}_{old_img['file_name']}"
                if old_filename in filename_to_new_id:
                    ann_copy = ann.copy()
                    ann_copy["image_id"] = filename_to_new_id[old_filename]
                    ann_copy["id"] = new_annotation_id
                    ann_copy["category_id"] = 1  # human_rgb
                    coco_output["annotations"].append(ann_copy)
                    new_annotation_id += 1

    # Process IR dataset
    for subfolder in ir_subfolders:
        dataset_path = Path(ir_dataset_path) / subfolder
        annotation_file = dataset_path / "_annotations.coco.json"

        if not annotation_file.exists():
            print(f"Annotation file not found in {dataset_path}")
            continue

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        category_map = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        orig_label = "Person"  # Adjust if your IR category name differs
        if orig_label not in category_map:
            print(f"Category '{orig_label}' not found in IR {subfolder} dataset")
            continue
        person_cat_id = category_map[orig_label]

        # Process images
        for img in coco_data["images"]:
            src_img_path = dataset_path / img["file_name"]
            dst_img_name = f"IR_{subfolder}_{img['file_name']}"
            dst_img_path = output_images / dst_img_name

            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Image {src_img_path} not found, skipping")
                continue

            # Assign new image ID
            filename_to_new_id[dst_img_name] = new_image_id
            img_copy = img.copy()
            img_copy["id"] = new_image_id
            img_copy["file_name"] = str(dst_img_path.relative_to(output_path))
            coco_output["images"].append(img_copy)
            new_image_id += 1

        # Process annotations
        for ann in coco_data["annotations"]:
            if ann["category_id"] == person_cat_id:
                old_img = next(i for i in coco_data["images"] if i["id"] == ann["image_id"])
                old_filename = f"IR_{subfolder}_{old_img['file_name']}"
                if old_filename in filename_to_new_id:
                    ann_copy = ann.copy()
                    ann_copy["image_id"] = filename_to_new_id[old_filename]
                    ann_copy["id"] = new_annotation_id
                    ann_copy["category_id"] = 0  # human_ir
                    coco_output["annotations"].append(ann_copy)
                    new_annotation_id += 1

    # Save the combined annotations
    output_annotation_file = output_path / "annotations.json"
    with open(output_annotation_file, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"Consolidated dataset saved to {output_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

def create_splits(data_root, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split the merged dataset into train, valid, and test sets, creating subfolders
    under data_root (data/merged) with images and _annotations.coco.json files.
    Ensure compatibility with the provided dataset configuration.
    
    Args:
        data_root (str): Path to merged dataset folder (data/merged)
        train_ratio (float): Proportion of images for training (default: 0.7)
        valid_ratio (float): Proportion of images for validation (default: 0.15)
        test_ratio (float): Proportion of images for testing (default: 0.15)
        random_seed (int): Seed for reproducibility
    """
    data_root = Path(data_root)
    merged_images = data_root / "images"
    merged_ann_file = data_root / "annotations.json"

    if not merged_images.exists() or not merged_ann_file.exists():
        raise FileNotFoundError(f"Merged images or annotations not found in {data_root}")

    # Load merged annotations
    with open(merged_ann_file, 'r') as f:
        coco_data = json.load(f)

    # Get all image files
    images = coco_data["images"]
    random.seed(random_seed)
    random.shuffle(images)

    # Calculate split sizes
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    valid_size = int(total_images * valid_ratio)
    test_size = total_images - train_size - valid_size  # Ensure all images are used

    # Split images
    train_images = images[:train_size]
    valid_images = images[train_size:train_size + valid_size]
    test_images = images[train_size + valid_size:]

    # Define splits
    splits = {
        "train": train_images,
        "valid": valid_images,
        "test": test_images
    }

    # Process each split
    for split_name, split_images in splits.items():
        split_dir = data_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Initialize COCO structure for this split
        split_coco = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "categories": coco_data["categories"],
            "images": [],
            "annotations": []
        }

        # Track new IDs
        new_image_id = 0
        new_annotation_id = 0
        old_to_new_id = {}  # Map old image ID to new image ID

        # Copy images and update image metadata
        for img in split_images:
            src_img_path = data_root / img["file_name"]
            dst_img_name = Path(img["file_name"]).name  # Keep filename (e.g., RGB_train_image1.jpg)
            dst_img_path = split_dir / dst_img_name

            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Image {src_img_path} not found, skipping")
                continue

            # Assign new image ID
            old_to_new_id[img["id"]] = new_image_id
            img_copy = img.copy()
            img_copy["id"] = new_image_id
            img_copy["file_name"] = f"{split_name}/{dst_img_name}"
            split_coco["images"].append(img_copy)
            new_image_id += 1

        # Update annotations
        for ann in coco_data["annotations"]:
            if ann["image_id"] in old_to_new_id:
                ann_copy = ann.copy()
                ann_copy["id"] = new_annotation_id
                ann_copy["image_id"] = old_to_new_id[ann["image_id"]]
                split_coco["annotations"].append(ann_copy)
                new_annotation_id += 1

        # Save split annotations
        split_ann_file = split_dir / "_annotations.coco.json"
        with open(split_ann_file, 'w') as f:
            json.dump(split_coco, f, indent=2)

        print(f"{split_name.capitalize()} split saved to {split_dir}")
        print(f"Images: {len(split_coco['images'])}, Annotations: {len(split_coco['annotations'])}")

def validate_coco_annotations(coco_file):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    image_ids = set(img["id"] for img in coco_data["images"])
    for ann in coco_data["annotations"]:
        if ann["image_id"] not in image_ids:
            print(f"Invalid annotation: image_id {ann['image_id']} not found in {coco_file}")
            return False
    print(f"Validation passed for {coco_file}")
    return True

def fixAnns(data_root='data/merged', splits=['train', 'valid', 'test']):
    """
    Remove split prefixes (e.g., 'train/', 'valid/', 'test/') from file_name in COCO annotation files.
    Backs up original files and verifies changes.
    
    Args:
        data_root (str): Root directory containing split folders (default: 'data/merged').
        splits (list): List of dataset splits to process (default: ['train', 'valid', 'test']).
    """
    data_root = Path(data_root)
    
    for split in splits:
        # Define paths
        ann_file = data_root / split / '_annotations.coco.json'
        backup_file = data_root / split / '_annotations_backup.coco.json'
        
        # Check if annotation file exists
        if not ann_file.exists():
            print(f"Warning: Annotation file {ann_file} not found. Skipping {split} split.")
            continue
        
        # Backup original file
        if not backup_file.exists():
            shutil.copy(ann_file, backup_file)
            print(f"Backed up {ann_file} to {backup_file}")
        
        # Load annotation file
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error: Failed to load {ann_file}: {e}")
            continue
        
        # Fix file_name entries
        modified = False
        for img in data['images']:
            old_filename = img['file_name']
            if old_filename.startswith(f'{split}/'):
                img['file_name'] = old_filename[len(f'{split}/'):]
                modified = True
                print(f"Updated {split} image: {old_filename} -> {img['file_name']}")
        
        # Save updated file
        if modified:
            try:
                with open(ann_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Successfully updated {ann_file}")
            except Exception as e:
                print(f"Error: Failed to save {ann_file}: {e}")
                continue
        else:
            print(f"No changes needed for {ann_file}")
        
        # Verify a few entries
        with open(ann_file, 'r') as f:
            data = json.load(f)
        for img in data['images'][:5]:
            if img['file_name'].startswith(f'{split}/'):
                print(f"Warning: Prefix {split}/ still present in {img['file_name']}")
            else:
                print(f"Verified: {img['file_name']} has no {split}/ prefix")

if __name__ == "__main__":
    rgb_dataset_path = "data/RGB"
    ir_dataset_path = "data/IR"
    output_path = "path/to/consolidated_dataset"
    mergeData(rgb_dataset_path, ir_dataset_path, output_path)
    data_root = "data/merged"
    create_splits(data_root, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_seed=42)
    for split in ["train", "valid", "test"]:
        validate_coco_annotations(data_root / split / "_annotations.coco.json")