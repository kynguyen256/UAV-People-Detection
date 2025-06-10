import os
import json
import logging
import argparse
import random
from pathlib import Path
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_predictions(json_path):
    """Load predictions from COCO JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded predictions from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load predictions: {str(e)}")
        raise

def visualize_images(predictions, img_dir, output_dir, num_images):
    """Visualize n random images with predicted bounding boxes using cv2."""
    output_dir = Path(output_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get random sample of images
    images = predictions['images']
    if num_images > len(images):
        logger.warning(f"Requested {num_images} images, but only {len(images)} available. Using all images.")
        num_images = len(images)
    selected_images = random.sample(images, num_images)
    
    # Category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in predictions['categories']}
    # Color mapping for categories (BGR format for OpenCV)
    cat_id_to_color = {0: (0, 0, 255),  # Red for human_ir
                       1: (255, 0, 0)}  # Blue for human_rgb
    
    for img_info in selected_images:
        try:
            img_id = img_info['id']
            img_name = img_info['file_name']
            img_path = Path(img_dir) / img_name
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            
            # Get annotations for this image
            annotations = [ann for ann in predictions['annotations'] if ann['image_id'] == img_id]
            
            # Draw bounding boxes and labels
            for ann in annotations:
                x, y, w, h = [int(coord) for coord in ann['bbox']]
                cat_id = ann['category_id']
                score = ann['score']
                
                # Draw rectangle
                color = cat_id_to_color.get(cat_id, (0, 255, 0))  # Default green if unexpected cat_id
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Add label with category and score
                label = f"{cat_id_to_name[cat_id]}: {score:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            save_path = output_dir / f"vis_{img_name}"
            cv2.imwrite(str(save_path), img)
            logger.info(f"Saved visualization to {save_path}")
            
        except Exception as e:
            logger.warning(f"Error processing image {img_name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Visualize random images with predictions from COCO JSON using cv2")
    parser.add_argument('--predictions-json', type=str, required=True, help="Path to predictions COCO JSON")
    parser.add_argument('--test-img-dir', type=str, required=True, help="Path to test images")
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory for visualizations")
    parser.add_argument('--num-images', type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()
    
    logger.info(f"Arguments: predictions_json={args.predictions_json}, test_img_dir={args.test_img_dir}, output_dir={args.output_dir}, num_images={args.num_images}")
    
    # Load predictions
    predictions = load_predictions(args.predictions_json)
    
    # Visualize images
    visualize_images(predictions, args.test_img_dir, output_dir, args.num_images)

if __name__ == "__main__":
    main()