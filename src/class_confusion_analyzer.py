import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

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

def analyze_class_confusion(gt_data, pred_data, conf_threshold=0.5, iou_threshold=0.5):
    """
    Analyze confusion between IR and RGB human classes.
    Returns detailed statistics about class confusions and their impact.
    """
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    num_classes = len(categories)
    
    # Initialize tracking structures
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track different types of errors
    errors_by_type = {
        'correct_detections': defaultdict(int),      # GT class X correctly predicted as X
        'class_confusions': defaultdict(int),         # GT class X predicted as Y
        'missed_detections': defaultdict(int),        # GT class X not detected
        'false_detections': defaultdict(int),         # No GT, predicted class X
        'low_iou_correct_class': defaultdict(int),    # Correct class but IoU < threshold
        'low_conf_correct_class': defaultdict(int)     # Correct class but conf < threshold
    }
    
    # Store examples of confusions for analysis
    confusion_examples = []
    
    # Group annotations by image
    gt_by_image = defaultdict(list)
    pred_by_image_all = defaultdict(list)  # All predictions
    pred_by_image_filtered = defaultdict(list)  # Above confidence threshold
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for ann in pred_data['annotations']:
        pred_by_image_all[ann['image_id']].append(ann)
        if ann['score'] >= conf_threshold:
            pred_by_image_filtered[ann['image_id']].append(ann)
    
    # Process each image
    for img_id in set(gt_by_image.keys()) | set(pred_by_image_all.keys()):
        gt_boxes = gt_by_image.get(img_id, [])
        pred_boxes_filtered = pred_by_image_filtered.get(img_id, [])
        pred_boxes_all = pred_by_image_all.get(img_id, [])
        
        # Match all predictions to ground truth (regardless of class)
        matched_gt = set()
        matched_pred = set()
        
        # Sort predictions by score
        pred_boxes_filtered_sorted = sorted(pred_boxes_filtered, 
                                          key=lambda x: x['score'], 
                                          reverse=True)
        
        # Match predictions to ground truth
        for pred_idx, pred in enumerate(pred_boxes_filtered_sorted):
            best_iou = 0
            best_gt_idx = -1
            best_gt = None
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt = gt
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # We have a match
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                
                gt_class = best_gt['category_id']
                pred_class = pred['category_id']
                
                confusion_matrix[gt_class, pred_class] += 1
                
                if gt_class == pred_class:
                    errors_by_type['correct_detections'][gt_class] += 1
                else:
                    errors_by_type['class_confusions'][(gt_class, pred_class)] += 1
                    confusion_examples.append({
                        'image_id': img_id,
                        'gt_class': categories[gt_class],
                        'pred_class': categories[pred_class],
                        'iou': best_iou,
                        'confidence': pred['score']
                    })
            else:
                # No match - this is a false positive
                errors_by_type['false_detections'][pred['category_id']] += 1
        
        # Check for missed ground truth boxes
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                gt_class = gt['category_id']
                
                # Check if there was a low-confidence correct prediction
                low_conf_match = False
                low_iou_match = False
                
                for pred in pred_boxes_all:
                    if pred['category_id'] == gt_class:
                        iou = compute_iou(pred['bbox'], gt['bbox'])
                        if iou >= iou_threshold and pred['score'] < conf_threshold:
                            low_conf_match = True
                            break
                        elif iou < iou_threshold and iou > 0.1 and pred['score'] >= conf_threshold:
                            low_iou_match = True
                
                if low_conf_match:
                    errors_by_type['low_conf_correct_class'][gt_class] += 1
                elif low_iou_match:
                    errors_by_type['low_iou_correct_class'][gt_class] += 1
                else:
                    errors_by_type['missed_detections'][gt_class] += 1
    
    return confusion_matrix, errors_by_type, confusion_examples, categories

def calculate_metrics_impact(errors_by_type, categories):
    """Calculate how different error types impact precision and recall."""
    metrics_impact = {}
    
    for class_id, class_name in categories.items():
        tp = errors_by_type['correct_detections'][class_id]
        
        # False positives include: wrong class predictions + pure false detections
        fp_false_detections = errors_by_type['false_detections'][class_id]
        fp_class_confusion = sum(count for (gt_class, pred_class), count 
                                in errors_by_type['class_confusions'].items() 
                                if pred_class == class_id and gt_class != class_id)
        fp_total = fp_false_detections + fp_class_confusion
        
        # False negatives include: missed detections + confused as other class + low conf/iou
        fn_missed = errors_by_type['missed_detections'][class_id]
        fn_class_confusion = sum(count for (gt_class, pred_class), count 
                               in errors_by_type['class_confusions'].items() 
                               if gt_class == class_id and pred_class != class_id)
        fn_low_conf = errors_by_type['low_conf_correct_class'][class_id]
        fn_low_iou = errors_by_type['low_iou_correct_class'][class_id]
        fn_total = fn_missed + fn_class_confusion + fn_low_conf + fn_low_iou
        
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0
        recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_impact[class_id] = {
            'tp': tp,
            'fp_total': fp_total,
            'fp_breakdown': {
                'false_detections': fp_false_detections,
                'class_confusion': fp_class_confusion
            },
            'fn_total': fn_total,
            'fn_breakdown': {
                'missed_detections': fn_missed,
                'class_confusion': fn_class_confusion,
                'low_confidence': fn_low_conf,
                'low_iou': fn_low_iou
            },
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics_impact

def plot_class_confusion_matrix(confusion_matrix, categories, conf_threshold, output_dir=None):
    """Plot the actual class confusion matrix."""
    class_names = [categories[i] for i in sorted(categories.keys())]
    
    plt.figure(figsize=(8, 6))
    
    # Create percentage version
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_pct = np.zeros_like(confusion_matrix, dtype=float)
    for i in range(len(categories)):
        if row_sums[i] > 0:
            confusion_matrix_pct[i] = confusion_matrix[i] / row_sums[i] * 100
    
    # Create annotation labels with both count and percentage
    labels = np.empty_like(confusion_matrix, dtype=object)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            count = confusion_matrix[i, j]
            pct = confusion_matrix_pct[i, j]
            labels[i, j] = f'{count}\n({pct:.1f}%)'
    
    sns.heatmap(confusion_matrix_pct, annot=labels, fmt='', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(f'Class Confusion Matrix (conf={conf_threshold:.2f})\nShowing counts and (percentages)', fontsize=14)
    
    # Add text about total samples
    total_gt = confusion_matrix.sum()
    plt.text(0.5, -0.12, f'Total matched detections: {total_gt}', 
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'class_confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class confusion matrix to {output_path}")
    
    plt.show()

def plot_error_breakdown(metrics_impact, categories, output_dir=None):
    """Plot breakdown of precision and recall losses by error type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    class_names = [categories[i] for i in sorted(categories.keys())]
    x = np.arange(len(class_names))
    width = 0.35
    
    # Precision impact (False Positives)
    fp_false_det = [metrics_impact[i]['fp_breakdown']['false_detections'] 
                    for i in sorted(categories.keys())]
    fp_class_conf = [metrics_impact[i]['fp_breakdown']['class_confusion'] 
                     for i in sorted(categories.keys())]
    
    ax1.bar(x, fp_false_det, width, label='False Detections', color='lightcoral')
    ax1.bar(x, fp_class_conf, width, bottom=fp_false_det, 
            label='Class Confusion', color='darkred')
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('False Positive Count', fontsize=12)
    ax1.set_title('Precision Loss Breakdown (False Positives)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Recall impact (False Negatives)
    fn_missed = [metrics_impact[i]['fn_breakdown']['missed_detections'] 
                 for i in sorted(categories.keys())]
    fn_class_conf = [metrics_impact[i]['fn_breakdown']['class_confusion'] 
                     for i in sorted(categories.keys())]
    fn_low_conf = [metrics_impact[i]['fn_breakdown']['low_confidence'] 
                   for i in sorted(categories.keys())]
    fn_low_iou = [metrics_impact[i]['fn_breakdown']['low_iou'] 
                  for i in sorted(categories.keys())]
    
    ax2.bar(x, fn_missed, width, label='Missed Detections', color='lightblue')
    bottom = np.array(fn_missed)
    ax2.bar(x, fn_class_conf, width, bottom=bottom, 
            label='Class Confusion', color='darkblue')
    bottom += np.array(fn_class_conf)
    ax2.bar(x, fn_low_conf, width, bottom=bottom, 
            label='Low Confidence', color='navy')
    bottom += np.array(fn_low_conf)
    ax2.bar(x, fn_low_iou, width, bottom=bottom, 
            label='Low IoU', color='midnightblue')
    
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('False Negative Count', fontsize=12)
    ax2.set_title('Recall Loss Breakdown (False Negatives)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'error_breakdown.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error breakdown to {output_path}")
    
    plt.show()

def plot_confusion_impact_on_metrics(metrics_impact, categories, output_dir=None):
    """Plot how class confusion specifically impacts precision and recall."""
    class_names = [categories[i] for i in sorted(categories.keys())]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (class_id, class_name) in enumerate(categories.items()):
        ax = axes[idx // 2, idx % 2]
        
        metrics = metrics_impact[class_id]
        
        # Calculate percentages
        fp_total = metrics['fp_total']
        fn_total = metrics['fn_total']
        
        if fp_total > 0:
            fp_conf_pct = metrics['fp_breakdown']['class_confusion'] / fp_total * 100
        else:
            fp_conf_pct = 0
            
        if fn_total > 0:
            fn_conf_pct = metrics['fn_breakdown']['class_confusion'] / fn_total * 100
        else:
            fn_conf_pct = 0
        
        # Create pie charts showing impact
        labels = []
        sizes = []
        colors = []
        
        if metrics['tp'] > 0:
            labels.append(f'True Positives\n({metrics["tp"]})')
            sizes.append(metrics['tp'])
            colors.append('lightgreen')
        
        if metrics['fp_breakdown']['class_confusion'] > 0:
            labels.append(f'FP: Class Confusion\n({metrics["fp_breakdown"]["class_confusion"]})')
            sizes.append(metrics['fp_breakdown']['class_confusion'])
            colors.append('orange')
        
        if metrics['fp_breakdown']['false_detections'] > 0:
            labels.append(f'FP: False Detections\n({metrics["fp_breakdown"]["false_detections"]})')
            sizes.append(metrics['fp_breakdown']['false_detections'])
            colors.append('lightcoral')
        
        if metrics['fn_breakdown']['class_confusion'] > 0:
            labels.append(f'FN: Class Confusion\n({metrics["fn_breakdown"]["class_confusion"]})')
            sizes.append(metrics['fn_breakdown']['class_confusion'])
            colors.append('darkblue')
        
        if metrics['fn_breakdown']['missed_detections'] > 0:
            labels.append(f'FN: Missed\n({metrics["fn_breakdown"]["missed_detections"]})')
            sizes.append(metrics['fn_breakdown']['missed_detections'])
            colors.append('lightblue')
        
        if metrics['fn_breakdown']['low_confidence'] > 0:
            labels.append(f'FN: Low Conf\n({metrics["fn_breakdown"]["low_confidence"]})')
            sizes.append(metrics['fn_breakdown']['low_confidence'])
            colors.append('navy')
        
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{class_name} Detection Breakdown\n' + 
                        f'Precision: {metrics["precision"]:.3f}, Recall: {metrics["recall"]:.3f}\n' +
                        f'Class confusion: {fp_conf_pct:.1f}% of FP, {fn_conf_pct:.1f}% of FN',
                        fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14)
            ax.set_title(f'{class_name} - No Data', fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'confusion_impact_breakdown.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion impact breakdown to {output_path}")
    
    plt.show()

def print_confusion_analysis(confusion_matrix, errors_by_type, metrics_impact, 
                           confusion_examples, categories):
    """Print detailed analysis of class confusion."""
    print("\n" + "="*70)
    print("CLASS CONFUSION ANALYSIS")
    print("="*70)
    
    # Overall confusion statistics
    total_detections = confusion_matrix.sum()
    diagonal_sum = np.trace(confusion_matrix)
    off_diagonal_sum = total_detections - diagonal_sum
    
    print(f"\nOverall Statistics:")
    print(f"  Total matched detections: {total_detections}")
    print(f"  Correct class predictions: {diagonal_sum} ({diagonal_sum/total_detections*100:.1f}%)")
    print(f"  Class confusions: {off_diagonal_sum} ({off_diagonal_sum/total_detections*100:.1f}%)")
    
    # Per-class confusion details
    print(f"\nClass Confusion Details:")
    for gt_class in categories:
        for pred_class in categories:
            if gt_class != pred_class and confusion_matrix[gt_class, pred_class] > 0:
                count = confusion_matrix[gt_class, pred_class]
                gt_total = confusion_matrix[gt_class].sum()
                pct = count / gt_total * 100 if gt_total > 0 else 0
                print(f"  {categories[gt_class]} → {categories[pred_class]}: " +
                      f"{count} times ({pct:.1f}% of {categories[gt_class]} detections)")
    
    # Impact on metrics
    print(f"\nImpact on Metrics:")
    for class_id, class_name in categories.items():
        metrics = metrics_impact[class_id]
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        
        if metrics['fp_total'] > 0:
            conf_pct = metrics['fp_breakdown']['class_confusion'] / metrics['fp_total'] * 100
            print(f"      - Class confusion causes {conf_pct:.1f}% of false positives")
        
        print(f"    Recall: {metrics['recall']:.3f}")
        
        if metrics['fn_total'] > 0:
            conf_pct = metrics['fn_breakdown']['class_confusion'] / metrics['fn_total'] * 100
            print(f"      - Class confusion causes {conf_pct:.1f}% of false negatives")
    
    # Example confusions
    if confusion_examples:
        print(f"\nExample Confusions (showing first 5):")
        for i, example in enumerate(confusion_examples[:5]):
            print(f"  {i+1}. Image {example['image_id']}: {example['gt_class']} → " +
                  f"{example['pred_class']} (conf={example['confidence']:.3f}, IoU={example['iou']:.3f})")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Analyze class confusion between IR and RGB humans')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--pred', required=True, help='Path to predictions JSON file')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots (default: current directory)')
    args = parser.parse_args()
    
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
    
    # Analyze class confusion
    logger.info(f"Analyzing class confusion (conf_thr={args.conf_threshold}, iou_thr={args.iou_threshold})")
    confusion_matrix, errors_by_type, confusion_examples, categories = analyze_class_confusion(
        gt_data, pred_data, args.conf_threshold, args.iou_threshold
    )
    
    # Calculate metrics impact
    metrics_impact = calculate_metrics_impact(errors_by_type, categories)
    
    # Print analysis
    print_confusion_analysis(confusion_matrix, errors_by_type, metrics_impact, 
                           confusion_examples, categories)
    
    # Generate visualizations
    plot_class_confusion_matrix(confusion_matrix, categories, args.conf_threshold, output_dir)
    plot_error_breakdown(metrics_impact, categories, output_dir)
    plot_confusion_impact_on_metrics(metrics_impact, categories, output_dir)

if __name__ == "__main__":
    main()
