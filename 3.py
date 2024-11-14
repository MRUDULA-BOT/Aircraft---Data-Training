import os
import shutil
import random

# Define the dataset directory and the output directories
dataset_dir = "D:/archive (1)/PlanesDataset"
train_dir = "D:/archive (1)/PlanesDataset/train"
val_dir = "D:/archive (1)/PlanesDataset/val"
test_dir = "D:/archive (1)/PlanesDataset/test"

# Create the directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all subdirectories (categories)
categories = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Split the images in each category
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle images for randomness
    random.shuffle(images)
    
    # Calculate the number of images for each split
    total_images = len(images)
    train_size = int(total_images * 0.7)
    val_size = int(total_images * 0.2)
    test_size = total_images - train_size - val_size
    
    # Split the images into train, val, and test
    train_images = images[:train_size]
    val_images = images[train_size:train_size+val_size]
    test_images = images[train_size+val_size:]
    
    # Move images into respective folders
    for img in train_images:
        shutil.move(os.path.join(category_path, img), os.path.join(train_dir, img))
    
    for img in val_images:
        shutil.move(os.path.join(category_path, img), os.path.join(val_dir, img))
    
    for img in test_images:
        shutil.move(os.path.join(category_path, img), os.path.join(test_dir, img))

print("Dataset split completed.")
