import os

def download_new_dataset():
    # Create a data/ directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the paths where the dataset should be located
    dataset_path = os.path.join(data_dir, "roboflow.zip")
    extracted_folder = os.path.join(data_dir, "roboflow_new_dataset")

    # Check if the dataset is already downloaded and extracted
    if not os.path.exists(extracted_folder):
        # Download the dataset using curl
        os.system(f'curl -L "https://universe.roboflow.com/ds/GCMBcNVKu5?key=BQqKlUUupX" > {dataset_path}')
        
        # Unzip the dataset
        os.system(f'unzip {dataset_path} -d {data_dir}')
        
        # Remove the ZIP file
        os.remove(dataset_path)

        print("Dataset downloaded and extracted successfully.")
    else:
        print("Dataset already downloaded and extracted.")

# Call the function to download the new dataset
download_new_dataset()
