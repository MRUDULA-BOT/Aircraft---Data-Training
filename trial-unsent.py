from ultralytics import YOLO
import os
import cv2

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

# Initialize and train YOLOv8 model
model = YOLO('yolov8n.pt')
model.train(
    data=yaml_path,
    epochs=10,
    batch=16,
    imgsz=640
)

# Load the best model for inference
best_model_path = 'runs/detect/train19/weights/best1.pt'
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