from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from datetime import datetime

def load_model(model_path):
    """Load the trained YOLO model"""
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_single_image(model, image_path, conf_threshold=0.25):
    """Evaluate model on a single image and return detailed metrics"""
    try:
        # Run inference
        start_time = time.time()
        results = model(image_path)[0]  # Get first result
        inference_time = time.time() - start_time
        
        # Get detections
        boxes = results.boxes
        
        # Process results
        detections = []
        if len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf)
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
                    class_id = int(box.cls)
                    detections.append({
                        'confidence': conf,
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return {
            'num_detections': len(detections),
            'detections': detections,
            'inference_time': inference_time
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def visualize_results(image_path, detections, output_path):
    """Visualize detections on the image and save it"""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Aircraft {conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save image
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error visualizing results: {e}")
        return False

def evaluate_model(model_path, test_folder, output_folder):
    """Complete evaluation of the model on a test dataset"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Prepare results storage
    all_results = []
    total_inference_time = 0
    total_images = 0
    
    # Process each image in the test folder
    for img_name in os.listdir(test_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_folder, img_name)
            
            # Evaluate image
            result = evaluate_single_image(model, img_path)
            if result:
                all_results.append({
                    'image': img_name,
                    'results': result
                })
                total_inference_time += result['inference_time']
                total_images += 1
                
                # Visualize and save results
                output_path = os.path.join(output_folder, f'detected_{img_name}')
                visualize_results(img_path, result['detections'], output_path)
    
    # Calculate summary statistics
    if total_images > 0:
        avg_inference_time = total_inference_time / total_images
        total_detections = sum(r['results']['num_detections'] for r in all_results)
        avg_detections = total_detections / total_images
        
        # Save summary report
        report_path = os.path.join(output_folder, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Test Images Processed: {total_images}\n")
            f.write(f"Average Inference Time: {avg_inference_time*1000:.2f}ms\n")
            f.write(f"Total Detections: {total_detections}\n")
            f.write(f"Average Detections per Image: {avg_detections:.2f}\n\n")
            
            f.write("Detailed Results:\n")
            f.write(f"{'-'*50}\n")
            for result in all_results:
                f.write(f"\nImage: {result['image']}\n")
                f.write(f"Detections: {result['results']['num_detections']}\n")
                f.write(f"Inference Time: {result['results']['inference_time']*1000:.2f}ms\n")
                
                for i, det in enumerate(result['results']['detections'], 1):
                    f.write(f"  Detection {i}:\n")
                    f.write(f"    Confidence: {det['confidence']:.2f}\n")
                    f.write(f"    Bounding Box: {[round(x, 2) for x in det['bbox']]}\n")
        
        print(f"\nEvaluation complete! Results saved to {output_folder}")
        print(f"Processed {total_images} images")
        print(f"Average inference time: {avg_inference_time*1000:.2f}ms")
        print(f"Total detections: {total_detections}")
        
    else:
        print("No images were processed successfully.")

# Usage example
if __name__ == "__main__":
    # Set your paths here
    MODEL_PATH = r"D:\Aircraft\yolov8n.pt"  # Path to your trained model
    TEST_FOLDER = r"D:\archive (1)\PlanesDataset\test\images"  # Folder containing test images
    OUTPUT_FOLDER = r"D:\archive (1)\evaluation_results"  # Where to save results
    
    evaluate_model(MODEL_PATH, TEST_FOLDER, OUTPUT_FOLDER)