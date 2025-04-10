import os
import json
import shutil
from pathlib import Path

def mergeData(rgb_dataset_path, ir_dataset_path, output_path):
    """
    Consolidate RGB and IR datasets into a single folder with unified COCO annotations.
    Assign category ID 0 to 'human_ir' (IR people) and ID 1 to 'human_rgb' (RGB people).
    Filter out non-person annotations.
    
    Args:
        rgb_dataset_path (str): Path to RGB dataset folder
        ir_dataset_path (str): Path to IR dataset folder
        output_path (str): Path to output consolidated dataset
    """
    # Create output directories
    output_path = Path(output_path)
    output_images = output_path / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    # Initialize new COCO annotation structure
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

    # Process each dataset
    for dataset_path, source_type, orig_label, new_label, category_id in [
        (rgb_dataset_path, "RGB", "humans", "human_rgb", 1),
        (ir_dataset_path, "IR", "People", "human_ir", 0)
    ]:
        dataset_path = Path(dataset_path)
        annotation_file = dataset_path / "_annotations.coco.json"

        if not annotation_file.exists():
            print(f"Annotation file not found in {dataset_path}")
            continue

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Create a mapping from category name to ID
        category_map = {cat["name"]: cat["id"] for cat in coco_data["categories"]}

        # Get person category ID
        if orig_label not in category_map:
            print(f"Category '{orig_label}' not found in {source_type} dataset")
            continue
        person_cat_id = category_map[orig_label]

        # Track image IDs to avoid duplicates
        image_id_offset = len(coco_output["images"])
        annotation_id_offset = len(coco_output["annotations"])

        # Process images
        for img in coco_data["images"]:
            # Copy image to output folder with prefixed filename
            src_img_path = dataset_path / img["file_name"]
            dst_img_name = f"{source_type}_{img['file_name']}"
            dst_img_path = output_images / dst_img_name

            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Image {src_img_path} not found, skipping")
                continue

            # Update image info
            img_copy = img.copy()
            img_copy["id"] += image_id_offset
            img_copy["file_name"] = str(dst_img_path.relative_to(output_path))
            coco_output["images"].append(img_copy)

        # Process annotations
        for ann in coco_data["annotations"]:
            # Only keep annotations for the person category
            if ann["category_id"] == person_cat_id:
                ann_copy = ann.copy()
                ann_copy["id"] += annotation_id_offset
                ann_copy["image_id"] += image_id_offset
                ann_copy["category_id"] = category_id
                coco_output["annotations"].append(ann_copy)

    # Save consolidated annotations
    output_annotation_file = output_path / "annotations.json"
    with open(output_annotation_file, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"Consolidated dataset saved to {output_path}")
    print(f"Images: {len(coco_output['images'])}")
    print(f"Annotations: {len(coco_output['annotations'])}")

if __name__ == "__main__":
    # Example paths (replace with your actual paths)
    rgb_dataset_path = "data/RGB"
    ir_dataset_path = "data/IR"
    output_path = "path/to/consolidated_dataset"

    mergeData(rgb_dataset_path, ir_dataset_path, output_path)
