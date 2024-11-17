# main_train.py

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.detr import create_detr_model

import torch
from torch.utils.data import DataLoader
import colorlog
from tqdm import tqdm

# Initialize logging
logger = colorlog.getLogger('training_logger')
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel('INFO')

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

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Progress bar for training
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch data
                pixel_values = batch['pixel_values']
                labels = batch['labels']

                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()

                # Log loss per batch
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        # Average training loss over all batches
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        # Save model checkpoint
        model_checkpoint_path = f"detr_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_checkpoint_path)
        logger.info(f"Model saved for epoch {epoch+1} as {model_checkpoint_path}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
                for batch_idx, batch in enumerate(valid_loader):
                    pixel_values = batch['pixel_values']
                    labels = batch['labels']

                    # Forward pass
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

                    # Log validation loss per batch
                    logger.info(f"Validation Batch {batch_idx+1}/{len(valid_loader)} - Loss: {outputs.loss.item():.4f}")

                    pbar.set_postfix(val_loss=val_loss / (pbar.n + 1))
                    pbar.update(1)

        avg_val_loss = val_loss / len(valid_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "detr_model.pth")
    logger.info("Final model saved as detr_model.pth")

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
        bbox = self.bboxes[idx]  # [x_min, y_min, width, height]
        label = self.labels[idx]
        
        # Calculate area (width * height)
        area = bbox[2] * bbox[3]
        
        # COCO-format annotation dictionary
        annotation = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": bbox,
                    "category_id": label,
                    "area": area,
                    "iscrowd": 0
                }
            ]
        }
        
        # Use the processor to prepare the inputs
        encoding = self.processor(images=image, annotations=[annotation], return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0)  # Remove batch dimension
        labels = encoding['labels'][0]  # Get the labels for this image
        
        return pixel_values, labels

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]  # Keep labels as a list of dicts
    return {'pixel_values': pixel_values, 'labels': labels}

if __name__ == '__main__':
    main()
