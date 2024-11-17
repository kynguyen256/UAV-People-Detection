from src.processing import check_coco_data
from src.download_data import download_data
from src.visualize import randomDisplay
from src.overlays import createOverlays, process_dataset
from src.heatmap import heatmaps

import os
import json
import glob

def main():
      
      # Download the data from Roboflow
      download_data()
      
      # Ensure the COCO dataset is correct
      check_coco_data()

      # Randomly display some images 
      randomDisplay()

      # Filter for Humans only
      process_dataset()
      
      # Overlay bounding boxes
      createOverlays()

      # create bb heatmap
      heatmaps()

if __name__ == "__main__":
    main()
