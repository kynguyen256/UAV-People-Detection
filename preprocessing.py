from src.processing import check_coco_data

import os
import json
import glob

def main():
    """
    Main function to check COCO data in specified directories.
    """
    # Directories to check
    directories = ['data/train', 'data/valid', 'data/test']
    
    for directory in directories:
        print(f"Checking directory: {directory}")
        check_coco_data(directory)

if __name__ == "__main__":
    main()
