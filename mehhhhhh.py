from ultralytics import YOLO
import os

# Set up paths for your dataset
dataset_path = r"D:\archive (1)\PlanesDataset"
train_images_path = os.path.join(dataset_path, "train")
val_images_path = os.path.join(dataset_path, "val")

# Write the data.yaml file for YOLOv8
data_yaml = {
    'train': train_images_path, 
    'val': val_images_path, 
    'nc': 20,  # Number of classes
    'names': ['aircraft'] * 20  # Assuming all 20 classes are 'aircraft'
}

yaml_path = os.path.join(dataset_path, 'data.yaml')
with open(yaml_path, 'w') as f:
    for key, value in data_yaml.items():
        f.write(f"{key}: {value}\n")

# Initialize YOLOv8 model (using yolov8 version)
model = YOLO('yolov8n.pt')  # This loads the base YOLOv8 model

# Train the model
model.train(
    data=yaml_path,  # Use the data.yaml file
    epochs=10,  # Training for 5 epochs
    batch=16,  # Batch size
    imgsz=640  # Image size for training
)

# Now that training is done, check if 'best.pt' exists
best_model_path = 'runs/train/exp/weights/best.pt'

# If the model file exists, load it; otherwise, print an error
if os.path.exists(best_model_path):
    # Load the best model for inference
    model = YOLO(best_model_path)  # Path to the best model
    
# Run inference on a new image
image_path = r"D:\Aircraft\images.jpg"  # Replace with your test image path
results = model(image_path)

# Display results with bounding boxes
result = results[0]

results.show()

# Optionally, save the results to a new file
results.save()  # Saves to 'runs/detect' folder

# Print out prediction results
print(f"Predictions: {results.pandas().xywh}")
