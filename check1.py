import os
import random
import shutil

# Paths to your dataset folder
dataset_path = "D:\\archive (1)\\PlanesDataset"  # Replace with your path
train_path = "D:\\archive (1)\\PlanesDataset\\train"
val_path = "D:\\archive (1)\\PlanesDataset\\val"
test_path = "D:\\archive (1)\\PlanesDataset\\test"

# Ensure that train, val, and test are folders, not files
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif os.path.isfile(directory):
        print(f"Oops! There is already a file named {directory}. Please rename or remove it.")

# Create directories for train, val, and test
create_directory(train_path)
create_directory(val_path)
create_directory(test_path)

# List all the image files (assuming images are .jpg or .png)
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

# Shuffle the image files to ensure random splitting
random.shuffle(image_files)

# Split the files
train_split = int(0.7 * len(image_files))  # 70% for training
val_split = int(0.15 * len(image_files))   # 15% for validation
test_split = len(image_files) - train_split - val_split  # 15% for testing

train_images = image_files[:train_split]
val_images = image_files[train_split:train_split + val_split]
test_images = image_files[train_split + val_split:]

# Function to copy images and labels to the respective folders
def copy_files(image_list, source_path, target_path):
    for image in image_list:
        # Copy the image file
        shutil.copy(os.path.join(source_path, image), os.path.join(target_path, image))
        
        # Copy the label file (same name but with .txt extension)
        label_file = image.replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(os.path.join(source_path, label_file)):
            shutil.copy(os.path.join(source_path, label_file), os.path.join(target_path, label_file))

# Copy the images and labels to the corresponding folders
copy_files(train_images, dataset_path, train_path)
copy_files(val_images, dataset_path, val_path)
copy_files(test_images, dataset_path, test_path)

print("Dataset has been split into train, val, and test folders!")
