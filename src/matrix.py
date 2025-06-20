import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Set up command-line argument parser
parser = argparse.ArgumentParser(description='Generate confusion matrix from COCO JSON files')
parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
parser.add_argument('--pred', required=True, help='Path to predictions JSON file')
args = parser.parse_args()

# Load COCO JSON files
with open(args.gt, 'r') as f:
    gt_data = json.load(f)
with open(args.pred, 'r') as f:
    pred_data = json.load(f)

# Extract category names for labels
categories = gt_data['categories']
category_names = {cat['id']: cat['name'] for cat in categories}  # {0: 'human_ir', 1: 'human_rgb'}
labels = [category_names[0], category_names[1]]  # ['human_ir', 'human_rgb']

# Extract category IDs from annotations, assuming annotations are matched by image_id
gt_labels = []
pred_labels = []

# Assuming annotations are aligned (same order and image_id)
for gt_ann, pred_ann in zip(gt_data['annotations'], pred_data['annotations']):
    assert gt_ann['image_id'] == pred_ann['image_id'], "Mismatched image_id in annotations"
    gt_labels.append(gt_ann['category_id'])
    pred_labels.append(pred_ann['category_id'])

# Ensure lengths match
assert len(gt_labels) == len(pred_labels), "Mismatch in number of annotations"

# Compute confusion matrix
cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1])

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix for human_ir vs human_rgb')
plt.tight_layout()
plt.show()

# Print raw confusion matrix
print("Confusion Matrix:")
print(cm)