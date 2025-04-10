from src.processing import check_coco_data
from src.download_RGB import download_RGB
from src.visualize import randomDisplay
from src.overlays import createOverlays, process_dataset
from src.heatmap import heatmaps
from src.download_IR import download_IR
from src.merger import mergeData

import os
import json
import glob

def main():
      
      # Download the data from Roboflow
      download_RGB()
      download_IR()
      
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

      mergeData()

if __name__ == "__main__":
    main()
