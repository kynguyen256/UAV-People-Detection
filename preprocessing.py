from src.processing import check_coco_data
from src.download_data import download_data
from src.visualize import randomDisplay

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

if __name__ == "__main__":
    main()
