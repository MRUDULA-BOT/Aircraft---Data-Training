from ultralytics import YOLO
import os
import cv2
from pathlib import Path

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

# Load YOLOv8 model (small or medium)
model = YOLO('yolov8n.pt')

# You can fine-tune it with additional augmentations and hyperparameters
model.train(
    data=yaml_path,
    epochs=40,  # Increase epochs to improve training
    batch=16,   # Adjust batch size depending on GPU
    imgsz=640,  # Image size, experiment with larger values if GPU supports it
    augment=True,  # Enable augmentations like flip, rotation, etc.
    optimizer="SGD",  # Try using SGD or Adam for different behavior
    lr0=0.01,  # Adjust learning rate if needed
    lrf=0.1,   # Learning rate schedule
    momentum=0.937,  # Momentum for optimization
    weight_decay=0.0005# L2 regularization to avoid overfitting
)

# Check if the model saved successfully and load it
best_model_path = 'runs/detect/train22/weights/last.pt'  # Update the path
if os.path.exists(best_model_path):
    model = YOLO(best_model_path)
else:
    print(f"Warning: Best model not found at {best_model_path}")

# Run inference on a new image
image_path = r"D:\Aircraft\images.jpg"
results = model(image_path)

# Display and save results
for i, result in enumerate(results):
    # Plot the results on the image
    boxes = result.boxes  # Boxes object for bbox outputs
    
    # Get the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Plot the detections
    annotated_img = result.plot()
    
    # Save the annotated image
    save_path = f"runs/detect/result_{i}.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    
    # Print detection details
    if len(boxes) > 0:
        print(f"\nDetections in image {i+1}:")
        for box in boxes:
            # Get confidence and class
            confidence = float(box.conf)
            class_id = int(box.cls)
            
            # Get bounding box coordinates (xywh format)
            x, y, w, h = box.xywh[0].tolist()
            
            print(f"Aircraft detected with {confidence:.2f} confidence")
            print(f"Bounding box: x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}")
    else:
        print(f"\nNo detections in image {i+1}")

print("\nResults have been saved to the 'runs/detect' directory")
