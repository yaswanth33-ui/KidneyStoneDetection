import matplotlib.pyplot as plt
from pathlib import Path
import os
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
from base64 import b64encode
import time

def predict(image):
    try:
        start_time = time.time()
        model_path = os.path.join(Path.cwd(), 'artifacts', 'best.pt')
        print(f"Loading model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        model = YOLO(model_path)
        print("Model loaded successfully")
        
        # Get original image dimensions
        image_dimensions = f"{image.size[0]}x{image.size[1]}"
        
        # Run prediction
        result = model.predict(image, verbose=False)
        print("Prediction completed")

        # Get the image with bounding boxes
        img_with_boxes = result[0].plot()

        # Convert the image with bounding boxes to a PIL Image
        img_with_boxes_pil = Image.fromarray(img_with_boxes)

        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        img_with_boxes_pil.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue()).decode()        
        names = result[0].names
        counts = {}
        boxes = result[0].boxes
        
        # Ensure result[0] exists and has the expected attributes
        if not hasattr(result[0], 'boxes') or not hasattr(result[0], 'names'):
            raise ValueError("Model response does not contain expected attributes 'boxes' or 'names'")

        # Validate 'boxes' and 'names' attributes
        if result[0].boxes is None or not isinstance(result[0].names, dict):
            raise ValueError("Invalid 'boxes' or 'names' in model response")

        # Initialize counts for all classes
        counts = {name: 0 for name in result[0].names.values()}
        
        # Initialize variables for kidney stone detection
        kidney_stone_detected = False
        max_confidence = 0.0
        prediction_text = "No kidney stones detected"
        total_detections = 0
        
        # Handle case when detections are found
        if boxes is not None and len(boxes) > 0:
            # Initialize counts for all classes
            for class_id in names.keys():
                counts[names[class_id]] = 0
            
            # Count detections and find max confidence
            for class_id in names.keys():
                class_mask = (boxes.cls == class_id)
                class_count = class_mask.sum().item()
                counts[names[class_id]] = class_count
                total_detections += class_count
                
                # Check for kidney stone detection and get highest confidence
                if class_count > 0:
                    class_confidences = boxes.conf[class_mask]
                    if len(class_confidences) > 0:
                        class_max_conf = float(class_confidences.max())
                        if class_max_conf > max_confidence:
                            max_confidence = class_max_conf
                        
                        # Mark as kidney stone detected and update prediction text
                        kidney_stone_detected = True
                        class_name = names[class_id]
                        prediction_text = f"Kidney stones detected - {class_name} (Count: {class_count})"
            
            # If multiple classes detected, show summary
            if total_detections > 0:
                detected_classes = [names[class_id] for class_id in names.keys() if counts[names[class_id]] > 0]
                if len(detected_classes) > 1:
                    prediction_text = f"Kidney stones detected - Multiple types found: {', '.join(detected_classes)}"
                    
        else:
            # No detections found - initialize counts to 0
            for class_id in names.keys():
                counts[names[class_id]] = 0
            print("No objects detected in the image")

        # Calculate processing time
        processing_time = f"{time.time() - start_time:.2f} seconds"
        
        print(f"Detection counts: {counts}")
        print(f"Max confidence: {max_confidence}")
        print(f"Processing time: {processing_time}")
        
        # Return results in the format expected by frontend
        return {
            'result_image': img_str,
            'prediction': prediction_text,
            'confidence': max_confidence,
            'counts': counts,
            'processing_time': processing_time,
            'image_dimensions': image_dimensions,
            'kidney_stones_detected': kidney_stone_detected
        }
        
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise e