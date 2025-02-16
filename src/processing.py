import os 
import glob
import json 
def verify_json(file_path):
    """
    Verifies if a given JSON file adheres to the COCO format.
    
    Args:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - tuple: A tuple containing a boolean indicating validity and a message.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # List of required fields in a COCO JSON file
        required_fields = ['images', 'annotations', 'categories']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return False, f"Missing '{field}' field"
        
        return True, "Valid COCO format"
    
    except json.JSONDecodeError as e:
        # If JSON is not properly formatted
        return False, f"JSON Decode Error: {e}"

def check_coco_data():
    """
    Checks all JSON files in a given directory (and its subdirectories) for validity.
    
    Raises:
    - ValueError: If the directory does not exist or no JSON files are found.
    """

    # Parent directory
    parent_dir = "data"

    # Directories to check (nested inside data/)
    directories = ['RGB/train', 'RGB/valid', 'RGB/test', 'IR/train', 'IR/valid']

    for directory in directories:
        full_path = os.path.join(parent_dir, directory)  # Ensure the correct full path
        print(f"Checking directory: {full_path}")
        check_coco(full_path)

# Call the function
if __name__ == "__main__":
    check_coco_data()


def check_coco(root_dir):
    
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory {root_dir} does not exist.")
    
    # Find all JSON files in the directory and subdirectories
    json_files = glob.glob(os.path.join(root_dir, '**/*.json'), recursive=True)
    if not json_files:
        raise ValueError(f"No JSON files found in {root_dir}")

    issues = []
    for json_file in json_files:
        exists = os.path.exists(json_file)
        if not exists:
            issues.append(f"File {json_file} does not exist.")
            continue
        
        valid, message = verify_json(json_file)
        if not valid:
            issues.append(f"File {json_file} is invalid: {message}")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("All files are valid and well-formatted.")
        

