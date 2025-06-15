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
    """Match predictions to ground truth based on IoU and category ID."""
    matched = []
    unmatched_preds = []
    unmatched_gts = gts.copy()
    
    for pred in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(unmatched_gts):
            if pred['category_id'] == gt['category_id']:  # Match only same category
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
    """Compute precision, recall, and mIoU for each confidence threshold for combined and per-class cases."""
    # Initialize metrics dictionary
    metrics = {'combined': {}, 'per_class': {}}
    classes = {0: 'human_ir', 1: 'human_rgb'}
    
    # Compute combined metrics (all classes)
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
    
    metrics['combined'] = {
        'precisions': precisions,
        'recalls': recalls,
        'mious': mious
    }
    
    # Compute per-class metrics
    for class_id in classes:
        class_preds = [p for p in preds if p['category_id'] == class_id]
        class_gts = [g for g in gts if g['category_id'] == class_id]
        
        precisions = []
        recalls = []
        mious = []
        
        for conf_thr in conf_thresholds:
            tp = 0
            fp = 0
            fn = 0
            iou_sum = 0
            num_matches = 0
            
            for img_id in set(p['image_id'] for p in class_preds):
                img_preds = [p for p in class_preds if p['image_id'] == img_id and p['score'] >= conf_thr]
                img_gts = [g for g in class_gts if g['image_id'] == img_id]
                
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
        
        metrics['per_class'][class_id] = {
            'precisions': precisions,
            'recalls': recalls,
            'mious': mious
        }
    
    return metrics

def plot_metrics(conf_thresholds, metrics, output_dir):
    """Generate and save combined plots for precision, recall, and mIoU for combined and per-class cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = {0: 'human_ir', 1: 'human_rgb'}
    
    # Plot combined metrics
    plt.figure(figsize=(10, 8))
    plt.plot(conf_thresholds, metrics['combined']['precisions'], label='Precision', color='b', linewidth=2)
    plt.plot(conf_thresholds, metrics['combined']['recalls'], label='Recall', color='g', linewidth=2)
    plt.plot(conf_thresholds, metrics['combined']['mious'], label='mIoU', color='r', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value')
    plt.title('Precision, Recall, and mIoU vs Confidence Threshold (Combined)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'metrics_vs_conf.png')
    plt.close()
    logger.info("Saved combined metrics plot for all classes")
    
    # Plot per-class metrics
    for class_id, class_name in classes.items():
        class_metrics = metrics['per_class'][class_id]
        plt.figure(figsize=(10, 8))
        plt.plot(conf_thresholds, class_metrics['precisions'], label='Precision', color='b', linewidth=2)
        plt.plot(conf_thresholds, class_metrics['recalls'], label='Recall', color='g', linewidth=2)
        plt.plot(conf_thresholds, class_metrics['mious'], label='mIoU', color='r', linewidth=2)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Metric Value')
        plt.title(f'Precision, Recall, and mIoU vs Confidence Threshold ({class_name})')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_dir / f'metrics_vs_conf_{class_name}.png')
        plt.close()
        logger.info(f"Saved combined metrics plot for {class_name}")

def process_images(model, image_paths, coco_gt, score_thr=0.1):
    """Run inference and collect predictions in COCO format for two classes."""
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
            logger.info(f"Processed {img_path.name}: {sum(len(res) for res in result)} detections")
            
            # Process predictions for both classes (human_ir: cat_id=0, human_rgb: cat_id=1)
            for class_idx, bboxes in enumerate(result):  # result[0] -> human_ir, result[1] -> human_rgb
                cat_id = class_idx  # Maps class_idx 0 to cat_id 0, class_idx 1 to cat_id 1
                for bbox in bboxes:
                    if bbox[4] >= score_thr:
                        x1, y1, x2, y2 = bbox[:4]
                        coco_results.append({
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(bbox[4])
                        })
        
        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    return coco_results, images_info

def save_coco_json(coco_results, images_info, output_json):
    """Save predictions in COCO JSON format, creating output directory if needed."""
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)  # Create output directory
    coco_json = {
        "images": images_info,
        "annotations": coco_results,
        "categories": [
            {"id": 0, "name": "human_ir", "supercategory": "person"},
            {"id": 1, "name": "human_rgb", "supercategory": "person"}
        ]
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
    metrics = compute_metrics(coco_results, coco_gt['annotations'], conf_thresholds)
    
    # Plot metrics
    plot_metrics(conf_thresholds, metrics, args.output_dir)

if __name__ == "__main__":
    main()