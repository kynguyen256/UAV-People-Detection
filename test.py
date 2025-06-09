import os
import json
import logging
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from google.colab import drive

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_coco_annotations(annotation_path):
    """Load COCO ground truth annotations."""
    try:
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        logger.info(f"Loaded annotations from {annotation_path}")
        return coco_data
    except Exception as e:
        logger.error(f"Failed to load annotations: {str(e)}")
        raise

def get_image_list(test_img_dir):
    """Retrieve list of image paths from the test image directory."""
    test_img_dir = Path(test_img_dir)
    if not test_img_dir.exists():
        logger.error(f"Test image directory does not exist: {test_img_dir}")
        raise FileNotFoundError(f"Test image directory does not exist: {test_img_dir}")
    
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [p for p in test_img_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    logger.info(f"Found {len(image_paths)} images in {test_img_dir}")
    return sorted(image_paths)

def load_model(config_path, checkpoint_path, device):
    """Initialize the CoDETR model."""
    try:
        model = init_detector(str(config_path), str(checkpoint_path), device=device)
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    inter_xmin = max(x1, x2)
    inter_ymin = max(y1, y2)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def match_predictions_to_gt(preds, gts, iou_thr=0.5):
    """Match predictions to ground truth based on IoU."""
    matched = []
    unmatched_preds = []
    unmatched_gts = gts.copy()
    
    for pred in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(unmatched_gts):
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_gt_idx >= 0:
            matched.append((pred, unmatched_gts[best_gt_idx], best_iou))
            unmatched_gts.pop(best_gt_idx)
        else:
            unmatched_preds.append(pred)
    
    return matched, unmatched_preds, unmatched_gts

def compute_metrics(preds, gts, conf_thresholds):
    """Compute precision, recall, and mIoU for each confidence threshold."""
    precisions = []
    recalls = []
    mious = []
    
    for conf_thr in conf_thresholds:
        tp = 0
        fp = 0
        fn = 0
        iou_sum = 0
        num_matches = 0
        
        for img_id in set(p['image_id'] for p in preds):
            img_preds = [p for p in preds if p['image_id'] == img_id and p['score'] >= conf_thr]
            img_gts = [g for g in gts if g['image_id'] == img_id]
            
            matched, unmatched_preds, unmatched_gts = match_predictions_to_gt(img_preds, img_gts)
            
            tp += len(matched)
            fp += len(unmatched_preds)
            fn += len(unmatched_gts)
            iou_sum += sum(iou for _, _, iou in matched)
            num_matches += len(matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        miou = iou_sum / num_matches if num_matches > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        mious.append(miou)
    
    return precisions, recalls, mious

def plot_metrics(conf_thresholds, precisions, recalls, mious, output_dir):
    """Generate and save separate plots for precision, recall, and mIoU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Precision Plot
    plt.figure(figsize=(8, 6))
    plt.plot(conf_thresholds, precisions, label='Precision', color='b')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Confidence Threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'precision_vs_conf.png')
    plt.close()
    logger.info("Saved precision plot")
    
    # Recall Plot
    plt.figure(figsize=(8, 6))
    plt.plot(conf_thresholds, recalls, label='Recall', color='g')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Confidence Threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'recall_vs_conf.png')
    plt.close()
    logger.info("Saved recall plot")
    
    # mIoU Plot
    plt.figure(figsize=(8, 6))
    plt.plot(conf_thresholds, mious, label='mIoU', color='r')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('mIoU')
    plt.title('mIoU vs Confidence Threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'miou_vs_conf.png')
    plt.close()
    logger.info("Saved mIoU plot")

def process_images(model, image_paths, coco_gt, score_thr=0.1):
    """Run inference and collect predictions in COCO format."""
    coco_results = []
    images_info = []
    image_id_map = {img['file_name']: img['id'] for img in coco_gt['images']}
    
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            
            height, width = img.shape[:2]
            if img_path.name not in image_id_map:
                logger.warning(f"No ground truth for image: {img_path.name}")
                continue
            
            image_id = image_id_map[img_path.name]
            images_info.append({
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            result = inference_detector(model, str(img_path))
            logger.info(f"Processed {img_path.name}: {len(result[0])} detections")
            
            bboxes = result[0]  # Single class
            for bbox in bboxes:
                if bbox[4] >= score_thr:
                    x1, y1, x2, y2 = bbox[:4]
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(bbox[4])
                    })
        
        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    return coco_results, images_info

def save_coco_json(coco_results, images_info, output_json):
    """Save predictions in COCO JSON format."""
    coco_json = {
        "images": images_info,
        "annotations": coco_results,
        "categories": [{"id": 1, "name": "person", "supercategory": "object"}]
    }
    try:
        with open(output_json, 'w') as f:
            json.dump(coco_json, f, indent=2)
        logger.info(f"COCO JSON saved to {output_json}")
    except Exception as e:
        logger.error(f"Failed to save COCO JSON: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate COCO predictions and metric plots for CoDETR")
    parser.add_argument('--config', type=str, required=True, help="Path to model config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--test-img-dir', type=str, required=True, help="Path to test images directory")
    parser.add_argument('--gt-json', type=str, required=True, help="Path to ground truth COCO JSON")
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory for plots and JSON")
    parser.add_argument('--score-thr', type=float, default=0.1, help="Minimum score threshold for detections")
    args = parser.parse_args()
    
    logger.info(f"Arguments: config={args.config}, checkpoint={args.checkpoint}, test_img_dir={args.test_img_dir}, gt_json={args.gt_json}, output_dir={args.output_dir}, score_thr={args.score_thr}")
        
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.config, args.checkpoint, device)
    
    # Load ground truth
    coco_gt = load_coco_annotations(args.gt_json)
    
    # Get image paths
    image_paths = get_image_list(args.test_img_dir)
    
    # Process images and get predictions
    coco_results, images_info = process_images(model, image_paths, coco_gt, args.score_thr)
    
    # Save COCO predictions
    save_coco_json(coco_results, images_info, Path(args.output_dir) / 'predictions.json')
    
    # Compute metrics for confidence thresholds
    conf_thresholds = np.linspace(0, 1, 100)
    precisions, recalls, mious = compute_metrics(coco_results, coco_gt['annotations'], conf_thresholds)
    
    # Plot metrics
    plot_metrics(conf_thresholds, precisions, recalls, mious, args.output_dir)

if __name__ == "__main__":
    main()