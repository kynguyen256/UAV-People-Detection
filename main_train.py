# main_train.py

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
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            pixel_values, target = batch
            outputs = model(pixel_values=pixel_values, labels=target)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Optionally, add validation steps here

    # Save model
    torch.save(model.state_dict(), "detr_model.pth")
    print("Model saved as detr_model.pth")

if __name__ == '__main__':
    main()
