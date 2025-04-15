import os
import json
import shutil
from pathlib import Path

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
    rgb_subfolders = ["train", "test", "val"]
    ir_subfolders = ["train", "val"]

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
        orig_label = "People"  # Adjust if your IR category name differs
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

if __name__ == "__main__":
    rgb_dataset_path = "data/RGB"
    ir_dataset_path = "data/IR"
    output_path = "path/to/consolidated_dataset"
    mergeData(rgb_dataset_path, ir_dataset_path, output_path)