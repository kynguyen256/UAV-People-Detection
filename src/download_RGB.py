import os
import os

def download_RGB():
    # Define the parent data directory
    parent_dir = "data"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Define the RGB directory inside data/
    data_dir = os.path.join(parent_dir, "RGB")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the paths for the extracted folders
    train_folder = os.path.join(data_dir, "train")
    test_folder = os.path.join(data_dir, "test")
    valid_folder = os.path.join(data_dir, "valid")

    # Check if the dataset is already extracted (i.e., train, test, and valid folders exist)
    if not (os.path.exists(train_folder) and os.path.exists(test_folder) and os.path.exists(valid_folder)):
        # Define the path for the dataset zip file
        dataset_path = os.path.join(data_dir, "roboflow.zip")

        # Check if the ZIP file is already downloaded
        if not os.path.exists(dataset_path):
            # Download the dataset using curl
            os.system(f'curl -L "https://universe.roboflow.com/ds/GCMBcNVKu5?key=BQqKlUUupX" > {dataset_path}')
            print("Dataset downloaded successfully.")

        # Unzip the dataset
        os.system(f'unzip {dataset_path} -d {data_dir}')
        
        # Remove the ZIP file
        os.remove(dataset_path)
        print("Dataset extracted successfully.")
    else:
        print("Dataset already exists with 'train', 'test', and 'valid' folders.")

# Call the function to download the new dataset
if __name__ == "__main__":
    download_RGB()
