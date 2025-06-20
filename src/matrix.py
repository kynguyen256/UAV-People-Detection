import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes based on IoU.
    Returns matches as (pred_idx, gt_idx, iou) tuples.
    """
    matches = []
    matched_gt = set()
    
    # Sort predictions by score in descending order
    pred_indices = sorted(range(len(pred_boxes)), 
                         key=lambda i: pred_boxes[i].get('score', 0), 
                         reverse=True)
    
    for pred_idx in pred_indices:
        pred = pred_boxes[pred_idx]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
                
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
    
    return matches

def generate_confusion_matrix(gt_data, pred_data, conf_threshold=0.5, iou_threshold=0.5):
    """
    Generate confusion matrix for object detection.
    
    For each ground truth box:
    - If matched with a prediction of the same class -> True Positive (diagonal)
    - If matched with a prediction of different class -> Confusion (off-diagonal)
    - If not matched -> False Negative (added to FN count)
    
    For each prediction:
    - If not matched -> False Positive (added to FP count)
    """
    # Get category info
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    num_classes = len(categories)
    
    # Initialize confusion matrix (num_classes x num_classes)
    # cm[i,j] = number of GT class i predicted as class j
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track false positives and false negatives per class
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    # Group annotations by image
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for ann in pred_data['annotations']:
        if ann['score'] >= conf_threshold:
            pred_by_image[ann['image_id']].append(ann)
    
    # Process each image
    for img_id in gt_by_image:
        gt_boxes = gt_by_image[img_id]
        pred_boxes = pred_by_image.get(img_id, [])
        
        # Match predictions to ground truth
        matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)
        
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # Process matches
        for pred_idx, gt_idx, iou in matches:
            gt_class = gt_boxes[gt_idx]['category_id']
            pred_class = pred_boxes[pred_idx]['category_id']
            
            cm[gt_class, pred_class] += 1
            
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
        
        # Count false negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt_indices:
                false_negatives[gt['category_id']] += 1
        
        # Count false positives (unmatched predictions)
        for pred_idx, pred in enumerate(pred_boxes):
            if pred_idx not in matched_pred_indices:
                false_positives[pred['category_id']] += 1
    
    return cm, false_positives, false_negatives, categories

def compute_metrics_for_threshold(args):
    """Compute metrics for a single confidence threshold (for parallel processing)."""
    conf_threshold, gt_data, pred_data, iou_threshold = args
    
    cm, fp, fn, categories = generate_confusion_matrix(
        gt_data, pred_data, conf_threshold, iou_threshold
    )
    
    # Calculate overall and per-class F1 scores
    f1_scores = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for class_id in categories:
        tp = cm[class_id, class_id]
        fp_count = fp[class_id]
        fn_count = fn[class_id]
        
        precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0
        recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[class_id] = f1
        overall_tp += tp
        overall_fp += fp_count
        overall_fn += fn_count
    
    # Calculate overall F1
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'threshold': conf_threshold,
        'overall_f1': overall_f1,
        'per_class_f1': f1_scores,
        'cm': cm,
        'fp': dict(fp),
        'fn': dict(fn)
    }

def find_optimal_threshold(gt_data, pred_data, iou_threshold=0.5, n_thresholds=50, n_jobs=None):
    """Find the confidence threshold that maximizes overall F1 score."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), n_thresholds)
    
    # Generate thresholds to test
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    logger.info(f"Evaluating {n_thresholds} confidence thresholds using {n_jobs} CPUs...")
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    args_list = [(thr, gt_data, pred_data, iou_threshold) for thr in thresholds]
    
    # Run parallel computation
    with Pool(n_jobs) as pool:
        results = pool.map(compute_metrics_for_threshold, args_list)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Threshold evaluation completed in {elapsed_time:.2f} seconds")
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['overall_f1'])
    
    return best_result, results

def plot_f1_vs_threshold(results, categories, best_threshold, output_dir=None):
    """Plot F1 scores vs confidence threshold."""
    thresholds = [r['threshold'] for r in results]
    overall_f1 = [r['overall_f1'] for r in results]
    
    # Per-class F1 scores
    per_class_f1 = defaultdict(list)
    for r in results:
        for class_id, f1 in r['per_class_f1'].items():
            per_class_f1[class_id].append(f1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot overall F1
    plt.plot(thresholds, overall_f1, 'k-', linewidth=2, label='Overall F1')
    
    # Plot per-class F1
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for i, (class_id, class_name) in enumerate(categories.items()):
        plt.plot(thresholds, per_class_f1[class_id], '--', color=colors[i], 
                label=f'{class_name} F1', alpha=0.7)
    
    # Mark best threshold
    best_f1 = max(overall_f1)
    plt.axvline(x=best_threshold, color='r', linestyle=':', alpha=0.7, 
               label=f'Best threshold ({best_threshold:.3f})')
    plt.axhline(y=best_f1, color='r', linestyle=':', alpha=0.3)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence Threshold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if output_dir:
        output_path = Path(output_dir) / 'f1_vs_threshold.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved F1 vs threshold plot to {output_path}")
    
    plt.show()

def plot_confusion_matrix(cm, categories, false_positives, false_negatives, conf_threshold, output_path=None):
    """Plot confusion matrix with additional FP/FN information."""
    class_names = [categories[i] for i in sorted(categories.keys())]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot main confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('True Class')
    ax1.set_title(f'Detection Confusion Matrix (conf={conf_threshold:.3f})')
    
    # Calculate and display metrics
    total_gt = cm.sum(axis=1) + np.array([false_negatives[i] for i in range(len(categories))])
    total_pred = cm.sum(axis=0) + np.array([false_positives[i] for i in range(len(categories))])
    
    # Create summary table
    summary_data = []
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = false_positives[i]
        fn = false_negatives[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        summary_data.append([class_name, tp, fp, fn, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
    
    # Plot summary table
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=summary_data,
                      colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Per-Class Metrics')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.show()

def print_detailed_metrics(cm, false_positives, false_negatives, categories, conf_threshold):
    """Print detailed metrics for each class."""
    print("\n" + "="*60)
    print(f"DETAILED DETECTION METRICS (Confidence Threshold: {conf_threshold:.3f})")
    print("="*60)
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for class_id, class_name in categories.items():
        tp = cm[class_id, class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        # Confusions with other classes
        confusions = []
        for other_id, other_name in categories.items():
            if other_id != class_id and cm[class_id, other_id] > 0:
                confusions.append(f"{other_name}: {cm[class_id, other_id]}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass: {class_name}")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        if confusions:
            print(f"  Confused with:   {', '.join(confusions)}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  F1-Score:        {f1:.4f}")
    
    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\nOVERALL METRICS:")
    print(f"  Total TP: {overall_tp}")
    print(f"  Total FP: {overall_fp}")
    print(f"  Total FN: {overall_fn}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1-Score:  {overall_f1:.4f}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix from COCO JSON files')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--pred', required=True, help='Path to predictions JSON file')
    
    # Confidence threshold options
    conf_group = parser.add_mutually_exclusive_group()
    conf_group.add_argument('--conf-threshold', type=float, default=None,
                           help='Fixed confidence threshold for predictions')
    conf_group.add_argument('--auto-threshold', action='store_true',
                           help='Automatically select threshold that maximizes F1')
    
    parser.add_argument('--n-thresholds', type=int, default=100,
                       help='Number of thresholds to evaluate when using --auto-threshold (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Number of CPUs to use for parallel processing (default: all available)')
    
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots (default: current directory)')
    parser.add_argument('--no-threshold-plot', action='store_true',
                       help='Skip plotting F1 vs threshold when using --auto-threshold')
    
    args = parser.parse_args()
    
    # Set default confidence threshold if not specified
    if args.conf_threshold is None and not args.auto_threshold:
        args.conf_threshold = 0.5
        logger.info(f"Using default confidence threshold: {args.conf_threshold}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO JSON files
    logger.info(f"Loading ground truth from: {args.gt}")
    with open(args.gt, 'r') as f:
        gt_data = json.load(f)
    
    logger.info(f"Loading predictions from: {args.pred}")
    with open(args.pred, 'r') as f:
        pred_data = json.load(f)
    
    # Determine confidence threshold
    if args.auto_threshold:
        best_result, all_results = find_optimal_threshold(
            gt_data, pred_data, args.iou_threshold, args.n_thresholds, args.n_jobs
        )
        
        conf_threshold = best_result['threshold']
        cm = best_result['cm']
        fp = best_result['fp']
        fn = best_result['fn']
        categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
        
        logger.info(f"Optimal confidence threshold: {conf_threshold:.3f} (F1: {best_result['overall_f1']:.4f})")
        
        # Plot F1 vs threshold unless disabled
        if not args.no_threshold_plot:
            plot_f1_vs_threshold(all_results, categories, conf_threshold, output_dir)
    else:
        # Use fixed threshold
        conf_threshold = args.conf_threshold
        logger.info(f"Using fixed confidence threshold: {conf_threshold}")
        cm, fp, fn, categories = generate_confusion_matrix(
            gt_data, pred_data, conf_threshold, args.iou_threshold
        )
    
    # Print detailed metrics
    print_detailed_metrics(cm, fp, fn, categories, conf_threshold)
    
    # Plot confusion matrix
    output_path = output_dir / f'confusion_matrix_conf{conf_threshold:.3f}.png'
    plot_confusion_matrix(cm, categories, fp, fn, conf_threshold, output_path)

if __name__ == "__main__":
    main()
