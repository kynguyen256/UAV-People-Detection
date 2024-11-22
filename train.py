# main_train.py

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.detr import create_detr_model

import colorlog
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize logging
logger = colorlog.getLogger('training_logger')
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel('INFO')


def box_cxcywh_to_xyxy(boxes):
    # Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def compute_iou_matrix(boxes1, boxes2):
    # boxes1: [N, 4], boxes2: [M, 4] in [x_min, y_min, x_max, y_max]
    # Compute the intersection over union between two sets of boxes
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # Expand boxes
    boxes1 = boxes1[:, None, :]  # [N, 1, 4]
    boxes2 = boxes2[None, :, :]  # [1, M, 4]

    # Compute intersection
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Compute union
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area  # [N, M]

    return iou


class MetricsCalculator:
    def __init__(self):
        self.reset_epoch_metrics()

    def reset_epoch_metrics(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.IoU_sum = 0.0
        self.num_TP = 0

    def update(self, pred_boxes, gt_boxes):
        # Convert boxes to xyxy format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes_xyxy, gt_boxes_xyxy)

        # For predicted boxes
        max_iou_per_pred, _ = iou_matrix.max(dim=1)
        TP_pred_mask = max_iou_per_pred > 0
        num_TP = TP_pred_mask.sum().item()
        num_FP = (~TP_pred_mask).sum().item()

        # For ground truth boxes
        max_iou_per_gt, _ = iou_matrix.max(dim=0)
        FN_gt_mask = max_iou_per_gt == 0
        num_FN = FN_gt_mask.sum().item()

        # For mIoU
        IoUs_TP = max_iou_per_pred[TP_pred_mask]
        sum_IoU_TP = IoUs_TP.sum().item()

        # Update metrics
        self.TP += num_TP
        self.FP += num_FP
        self.FN += num_FN
        self.IoU_sum += sum_IoU_TP
        self.num_TP += num_TP

    def compute_metrics(self):
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
        mIoU = self.IoU_sum / self.num_TP if self.num_TP > 0 else 0
        return precision, recall, mIoU


class Plotter:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_mIoUs = []
        self.val_mIoUs = []

    def update_train_metrics(self, loss, precision, recall, mIoU):
        self.train_losses.append(loss)
        self.train_precisions.append(precision)
        self.train_recalls.append(recall)
        self.train_mIoUs.append(mIoU)

    def update_val_metrics(self, loss, precision, recall, mIoU):
        self.val_losses.append(loss)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_mIoUs.append(mIoU)

    def plot_metrics(self, num_epochs):
        epochs_list = list(range(1, num_epochs + 1))

        # Plot for precision, recall, mIoU
        plt.figure()
        plt.plot(epochs_list, self.train_precisions, label='Train Precision')
        plt.plot(epochs_list, self.val_precisions, label='Val Precision')
        plt.plot(epochs_list, self.train_recalls, label='Train Recall')
        plt.plot(epochs_list, self.val_recalls, label='Val Recall')
        plt.plot(epochs_list, self.train_mIoUs, label='Train mIoU')
        plt.plot(epochs_list, self.val_mIoUs, label='Val mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Precision, Recall, mIoU over Epochs')
        plt.legend()
        plt.savefig('metrics_plot.png')
        plt.close()

        # Plot for loss
        plt.figure()
        plt.plot(epochs_list, self.train_losses, label='Train Loss')
        plt.plot(epochs_list, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()


class Trainer:
    def __init__(self, model, processor, train_loader, val_loader, optimizer, num_epochs):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.plotter = Plotter()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        metrics_calculator = MetricsCalculator()

        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs} - Training") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch data to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch['labels'].items()}

                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update metrics
                batch_size = pixel_values.shape[0]
                pred_boxes = outputs.pred_boxes
                
                for i in range(batch_size):
                    pred_boxes_i = pred_boxes[i].detach().cpu()
                    labels_i = labels[i]
                    gt_boxes_i = labels_i['boxes'].cpu()
                    
                    metrics_calculator.update(pred_boxes_i, gt_boxes_i)

                # Compute batch metrics
                precision, recall, mIoU = metrics_calculator.compute_metrics()
                
                # Log progress
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))
                pbar.update(1)

                if batch_idx % 10 == 0:  # Log every 10 batches
                    logger.info(f"Batch {batch_idx+1}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

        # Compute epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        precision, recall, mIoU = metrics_calculator.compute_metrics()
        
        return avg_loss, precision, recall, mIoU

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        metrics_calculator = MetricsCalculator()

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation") as pbar:
                for batch_idx, batch in enumerate(self.val_loader):
                    # Move batch data to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch['labels'].items()}

                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

                    # Update metrics
                    batch_size = pixel_values.shape[0]
                    pred_boxes = outputs.pred_boxes

                    for i in range(batch_size):
                        pred_boxes_i = pred_boxes[i].cpu()
                        labels_i = labels[i]
                        gt_boxes_i = labels_i['boxes'].cpu()
                        
                        metrics_calculator.update(pred_boxes_i, gt_boxes_i)

                    pbar.set_postfix(val_loss=val_loss / (batch_idx + 1))
                    pbar.update(1)

        avg_loss = val_loss / len(self.val_loader)
        precision, recall, mIoU = metrics_calculator.compute_metrics()
        
        return avg_loss, precision, recall, mIoU

class COCOCustomDataset(Dataset):
    def __init__(self, annotation_file, images_dir, processor):
        self.processor = processor
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.image_files = []
        self.bboxes = []
        self.labels = []

        # Map image IDs to filenames
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        anns_per_image = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in anns_per_image:
                anns_per_image[image_id] = []
            anns_per_image[image_id].append(ann)

        for image_id, anns in anns_per_image.items():
            file_name = id_to_filename[image_id]
            image_path = os.path.join(images_dir, file_name)
            if not os.path.exists(image_path):
                continue

            self.image_files.append(image_path)
            bboxes = [ann['bbox'] for ann in anns]
            labels = [1 for _ in anns]  # Single class
            self.bboxes.append(bboxes)
            self.labels.append(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        bboxes = self.bboxes[idx]  # List of [x_min, y_min, width, height]
        labels = self.labels[idx]
    
        # Calculate area
        areas = [bbox[2] * bbox[3] for bbox in bboxes]
    
        # COCO-format annotations
        annotations_list = []
        for bbox, label, area in zip(bboxes, labels, areas):
            annotations_list.append({
                "bbox": bbox,
                "category_id": label,
                "area": area,
                "iscrowd": 0
            })
    
        # Prepare annotations dict with 'image_id' and 'annotations'
        annotations_dict = {
            "image_id": idx,  # Include image_id
            "annotations": annotations_list
        }
    
        # Prepare inputs using the processor
        encoding = self.processor(images=image, annotations=annotations_dict, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0)  # Remove batch dimension
        target = encoding['labels'][0]  # Get the labels for this image
    
        return pixel_values, target

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]  # Keep labels as a list of dicts
    return {'pixel_values': pixel_values, 'labels': labels}


def main():
    # Paths
    train_images_dir = 'data/train'
    train_annotation_file = 'data/train/train_annotations.coco.json'
    valid_images_dir = 'data/valid'
    valid_annotation_file = 'data/valid/valid_annotations.coco.json'

    # Load model and processor
    model, processor = create_detr_model()
    model.train()

    # Dataset and DataLoader
    train_dataset = COCOCustomDataset(train_annotation_file, train_images_dir, processor)
    valid_dataset = COCOCustomDataset(valid_annotation_file, valid_images_dir, processor)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=collate_fn)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10

    # Create Trainer and start training
    trainer = Trainer(model, processor, train_loader, valid_loader, optimizer, num_epochs)
    trainer.train()


if __name__ == '__main__':
    main()
