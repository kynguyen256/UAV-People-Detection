from src.processing import check_coco_data
from src.download_RGB import download_RGB
from src.visualize import randomDisplay
from src.overlays import createOverlays, process_dataset
from src.heatmap import heatmaps
from src.download_IR import download_IR
from src.merger import mergeData, create_splits, validate_coco_annotations

def main():
      
      # Download the data from Roboflow
      download_RGB()
      download_IR()
      
      # Ensure the COCO dataset is correct
      #check_coco_data()

      # Randomly display some images 
      #randomDisplay()

      # Filter for Humans only
      #process_dataset()
      
      # Overlay bounding boxes
      #createOverlays()

      # create bb heatmap
      #heatmaps()

      rgb_dataset_path = "data/RGB"
      ir_dataset_path = "data/IR"
      output_path = "data/merged"

      mergeData(rgb_dataset_path, ir_dataset_path, output_path)

      data_root = "data/merged"
      create_splits(data_root, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_seed=42)
      for split in ["train", "valid", "test"]:
        validate_coco_annotations(data_root / split / "_annotations.coco.json")

if __name__ == "__main__":
    main()
