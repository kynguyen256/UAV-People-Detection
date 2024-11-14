# main_test.py

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.detr import create_detr_model

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
        for ann in data['annotations']:
            image_id = ann['image_id']
            file_name = id_to_filename[image_id]
            image_path = os.path.join(images_dir, file_name)
            if not os.path.exists(image_path):
                continue
            
            self.image_files.append(image_path)
            bbox = ann['bbox']
            self.bboxes.append(bbox)
            self.labels.append(1)  # Single class

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        bbox = self.bboxes[idx]
        label = self.labels[idx]
        encoding = self.processor(images=image, annotations={"boxes": [bbox], "labels": [label]}, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()  # (3, height, width)
        target = {
            "boxes": encoding['labels']['boxes'],
            "labels": encoding['labels']['labels']
        }
        return pixel_values, target

def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, target in data_loader:
            pixel_values = pixel_values.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            outputs = model(pixel_values=pixel_values, labels=target)
            loss = outputs.loss
            total_loss += loss.item()

            # Calculate accuracy based on predictions
            pred_labels = outputs.logits.argmax(dim=-1)
            true_labels = target["labels"]
            correct += (pred_labels == true_labels).sum().item()
            total += true_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

def main():
    # Paths
    valid_images_dir = 'data/valid'
    valid_annotation_file = 'data/valid/valid_annotations.coco.json'

    # Load model and processor
    model, processor = create_detr_model()
    model.load_state_dict(torch.load("detr_model.pth"))  # Load trained weights
    model.to("cuda")

    # Dataset and DataLoader
    valid_dataset = COCOCustomDataset(valid_annotation_file, valid_images_dir, processor)
    valid_loader = DataLoader(valid_dataset, batch_size=4)

    # Evaluate the model
    evaluate_model(model, valid_loader, device="cuda")

if __name__ == '__main__':
    main()
