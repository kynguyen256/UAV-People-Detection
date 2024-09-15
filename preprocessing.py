from src.processing import check_coco_data
from src.download_data import download_data

import os
import json
import glob

def main():

      # 
    
      # Download the data from Roboflow
      download_data()

if __name__ == "__main__":
    main()
