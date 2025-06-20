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

def compute_per_class_metrics(gt_data, pred_data, class_id, conf_threshold=0.5, iou_threshold=0.5):
    """
    Compute binary confusion matrix for a specific class.
    Treats the problem as binary: class_id vs everything else (including no detection).
    """
    tp = 0  # Ground truth is class_id, prediction is class_id
    fp = 0  # No ground truth or ground truth is different class, prediction is class_id
    fn = 0  # Ground truth is class_id, no matching prediction or prediction is different class
    
    # Group annotations by image
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for ann in pred_data['annotations']:
        if ann['score'] >= conf_threshold:
            pred_by_image[ann['image_id']].append(ann)
    
    # Process each image
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    
    for img_id in all_image_ids:
        gt_boxes = gt_by_image.get(img_id, [])
        pred_boxes = pred_by_image.get(img_id, [])
        
        # Get class-specific boxes
        gt_class_boxes = [gt for gt in gt_boxes if gt['category_id'] == class_id]
        pred_class_boxes = [pred for pred in pred_boxes if pred['category_id'] == class_id]
        
        # Match predictions to ground truth (only for same class)
        matches = match_predictions_to_gt(pred_class_boxes, gt_class_boxes, iou_threshold)
        
        # Count true positives
        tp += len(matches)
        
        # Count false negatives (unmatched ground truth of this class)
        matched_gt_indices = {gt_idx for _, gt_idx, _ in matches}
        fn += len([gt for i, gt in enumerate(gt_class_boxes) if i not in matched_gt_indices])
        
        # Count false positives (unmatched predictions of this class)
        matched_pred_indices = {pred_idx for pred_idx, _, _ in matches}
        fp += len([pred for i, pred in enumerate(pred_class_boxes) if i not in matched_pred_indices])
    
    return tp, fp, fn

def compute_metrics_for_threshold_per_class(args):
    """Compute metrics for a single confidence threshold for a specific class."""
    conf_threshold, gt_data, pred_data, class_id, iou_threshold = args
    
    tp, fp, fn = compute_per_class_metrics(
        gt_data, pred_data, class_id, conf_threshold, iou_threshold
    )
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': conf_threshold,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def find_optimal_threshold_per_class(gt_data, pred_data, class_id, iou_threshold=0.5, n_thresholds=50, n_jobs=None):
    """Find the confidence threshold that maximizes F1 score for a specific class."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), n_thresholds)
    
    # Generate thresholds to test
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    logger.info(f"Evaluating {n_thresholds} confidence thresholds for class {class_id} using {n_jobs} CPUs...")
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    args_list = [(thr, gt_data, pred_data, class_id, iou_threshold) for thr in thresholds]
    
    # Run parallel computation
    with Pool(n_jobs) as pool:
        results = pool.map(compute_metrics_for_threshold_per_class, args_list)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Threshold evaluation for class {class_id} completed in {elapsed_time:.2f} seconds")
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['f1'])
    
    return best_result, results

def plot_per_class_confusion_matrices(results_per_class, categories, output_dir=None):
    """Plot confusion matrices for each class side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (class_id, class_name) in enumerate(categories.items()):
        best_result = results_per_class[class_id]['best']
        
        # Create confusion matrix data
        tp = best_result['tp']
        fp = best_result['fp']
        fn = best_result['fn']
        tn = "N/A"  # True negatives not applicable in object detection
        
        # Create a simple 2x2 visualization
        ax = axes[idx]
        
        # Create custom confusion matrix visualization
        data = [[tp, fp], [fn, tn]]
        labels = [[f"TP\n{tp}", f"FP\n{fp}"], [f"FN\n{fn}", "TN\n(N/A)"]]
        
        # Create heatmap with custom colormap
        mask = np.array([[False, False], [False, True]])  # Mask TN
        cmap = sns.color_palette("Blues", as_cmap=True)
        
        sns.heatmap(data, annot=labels, fmt='', cmap=cmap, mask=mask,
                   cbar=False, square=True, linewidths=2, linecolor='black',
                   annot_kws={'size': 14}, ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xticklabels([f'{class_name}', 'Not ' + class_name], fontsize=10)
        ax.set_yticklabels([f'{class_name}', 'Not ' + class_name], fontsize=10)
        
        # Add metrics text
        precision = best_result['precision']
        recall = best_result['recall']
        f1 = best_result['f1']
        threshold = best_result['threshold']
        
        metrics_text = f"Threshold: {threshold:.3f}\n"
        metrics_text += f"Precision: {precision:.3f}\n"
        metrics_text += f"Recall: {recall:.3f}\n"
        metrics_text += f"F1-Score: {f1:.3f}"
        
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        ax.set_title(f'{class_name} Detection Performance', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'per_class_confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class confusion matrices to {output_path}")
    
    plt.show()

def plot_f1_vs_threshold_per_class(results_per_class, categories, output_dir=None):
    """Plot F1, precision, and recall vs threshold for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (class_id, class_name) in enumerate(categories.items()):
        all_results = results_per_class[class_id]['all']
        best_threshold = results_per_class[class_id]['best']['threshold']
        
        thresholds = [r['threshold'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        
        ax = axes[idx]
        
        # Plot metrics
        ax.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
        ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        ax.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        
        # Mark best threshold
        best_f1 = max(f1_scores)
        ax.axvline(x=best_threshold, color='k', linestyle=':', alpha=0.7, 
                  label=f'Best threshold ({best_threshold:.3f})')
        
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{class_name} Metrics vs Confidence Threshold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'metrics_vs_threshold_per_class.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class metrics vs threshold plot to {output_path}")
    
    plt.show()

def plot_combined_f1_comparison(results_per_class, categories, output_dir=None):
    """Plot F1 scores for both classes on the same plot for comparison."""
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red']
    
    for idx, (class_id, class_name) in enumerate(categories.items()):
        all_results = results_per_class[class_id]['all']
        best_threshold = results_per_class[class_id]['best']['threshold']
        
        thresholds = [r['threshold'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        
        plt.plot(thresholds, f1_scores, color=colors[idx], linewidth=2, 
                label=f'{class_name} F1')
        
        # Mark best threshold
        best_f1 = max(f1_scores)
        plt.axvline(x=best_threshold, color=colors[idx], linestyle=':', alpha=0.7, 
                   label=f'{class_name} best ({best_threshold:.3f}, F1={best_f1:.3f})')
    
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Comparison Across Classes', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'f1_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved F1 comparison plot to {output_path}")
    
    plt.show()

def print_per_class_metrics(results_per_class, categories):
    """Print detailed metrics for each class."""
    print("\n" + "="*70)
    print("PER-CLASS DETECTION METRICS (Optimal Thresholds)")
    print("="*70)
    
    for class_id, class_name in categories.items():
        best = results_per_class[class_id]['best']
        
        print(f"\nClass: {class_name}")
        print(f"  Optimal Threshold: {best['threshold']:.3f}")
        print(f"  True Positives:    {best['tp']}")
        print(f"  False Positives:   {best['fp']}")
        print(f"  False Negatives:   {best['fn']}")
        print(f"  Precision:         {best['precision']:.4f}")
        print(f"  Recall:            {best['recall']:.4f}")
        print(f"  F1-Score:          {best['f1']:.4f}")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Generate per-class confusion matrices from COCO JSON files')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--pred', required=True, help='Path to predictions JSON file')
    
    # Confidence threshold options
    conf_group = parser.add_mutually_exclusive_group()
    conf_group.add_argument('--conf-threshold', type=float, default=None,
                           help='Fixed confidence threshold for all classes')
    conf_group.add_argument('--auto-threshold', action='store_true',
                           help='Automatically select optimal threshold per class')
    
    parser.add_argument('--n-thresholds', type=int, default=50,
                       help='Number of thresholds to evaluate when using --auto-threshold (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Number of CPUs to use for parallel processing (default: all available)')
    
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots (default: current directory)')
    parser.add_argument('--no-threshold-plot', action='store_true',
                       help='Skip plotting metrics vs threshold')
    
    args = parser.parse_args()
    
    # Set default behavior to auto-threshold if not specified
    if args.conf_threshold is None and not args.auto_threshold:
        args.auto_threshold = True
        logger.info("No threshold specified, using --auto-threshold by default")
    
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
    
    # Get categories
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Process each class
    results_per_class = {}
    
    if args.auto_threshold:
        # Find optimal threshold for each class
        for class_id, class_name in categories.items():
            logger.info(f"\nProcessing class: {class_name}")
            best_result, all_results = find_optimal_threshold_per_class(
                gt_data, pred_data, class_id, args.iou_threshold, 
                args.n_thresholds, args.n_jobs
            )
            
            results_per_class[class_id] = {
                'best': best_result,
                'all': all_results
            }
            
            logger.info(f"Class {class_name}: Optimal threshold = {best_result['threshold']:.3f}, F1 = {best_result['f1']:.4f}")
    
    else:
        # Use fixed threshold for all classes
        conf_threshold = args.conf_threshold
        logger.info(f"Using fixed confidence threshold: {conf_threshold}")
        
        for class_id, class_name in categories.items():
            tp, fp, fn = compute_per_class_metrics(
                gt_data, pred_data, class_id, conf_threshold, args.iou_threshold
            )
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results_per_class[class_id] = {
                'best': {
                    'threshold': conf_threshold,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'all': None  # No threshold search performed
            }
    
    # Print detailed metrics
    print_per_class_metrics(results_per_class, categories)
    
    # Generate visualizations
    plot_per_class_confusion_matrices(results_per_class, categories, output_dir)
    
    if args.auto_threshold and not args.no_threshold_plot:
        plot_f1_vs_threshold_per_class(results_per_class, categories, output_dir)
        plot_combined_f1_comparison(results_per_class, categories, output_dir)

if __name__ == "__main__":
    main()
