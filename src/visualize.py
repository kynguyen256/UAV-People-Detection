import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def randomDisplay():
  
  # Define the path to your folder
  folder_path = '/content/UAV-People-Detection/data/train/'  # Replace with your folder path
  
  # Get a list of image files in the folder
  files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
  
  # Define how many images you want to sample
  sample_size = 5  # Adjust this number to your preference
  
  # Randomly select images from the list
  sampled_files = random.sample(files, sample_size)
  
  # Display the sampled images
  for file_name in sampled_files:
      image_path = os.path.join(folder_path, file_name)
      img = mpimg.imread(image_path)
      plt.imshow(img)
      plt.axis('off')  # Hide axes
      plt.show()
  
