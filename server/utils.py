import matplotlib.pyplot as plt
from pathlib import Path
import os
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
from base64 import b64encode
import time
import gc
import torch

# Global model variable to load once and reuse
_model = None

def get_model():
    """Load model only once and reuse it"""
    global _model
    if _model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'artifacts', 'best.pt')
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load model with CPU-only optimizations for memory
        _model = YOLO(model_path)
        
        # Force CPU usage and disable model optimization that causes issues
        _model.to('cpu')
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Model loaded and cached successfully")
    
    return _model

def predict(image):
    try:
        start_time = time.time()
        
        # Resize image if too large to save memory
        max_size = 640  # YOLO standard input size
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Image resized to: {image.size}")
        
        # Get original image dimensions
        image_dimensions = f"{image.size[0]}x{image.size[1]}"
        
        # Get cached model
        model = get_model()
        print("Using cached model")
        
        # Run prediction with optimizations
        with torch.no_grad():  # Disable gradient computation to save memory
            result = model.predict(
                image, 
                verbose=False,
                save=False,  # Don't save intermediate files
                show=False,  # Don't show plots
                conf=0.25,   # Lower confidence threshold for faster processing
                max_det=50,  # Limit max detections
                device='cpu',  # Force CPU usage
                half=False,  # Disable half precision to avoid issues
                augment=False,  # Disable test-time augmentation
                agnostic_nms=False,  # Disable class-agnostic NMS
                retina_masks=False  # Disable retina masks
            )
        print("Prediction completed")

        # Get the image with bounding boxes
        img_with_boxes = result[0].plot()

        # Convert the image with bounding boxes to a PIL Image
        img_with_boxes_pil = Image.fromarray(img_with_boxes)

        # Resize output image to reduce base64 size
        if max(img_with_boxes_pil.size) > 800:
            ratio = 800 / max(img_with_boxes_pil.size)
            new_size = tuple(int(dim * ratio) for dim in img_with_boxes_pil.size)
            img_with_boxes_pil = img_with_boxes_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Convert PIL Image to base64 string with compression
        buffered = io.BytesIO()
        img_with_boxes_pil.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = b64encode(buffered.getvalue()).decode()
        
        names = result[0].names
        boxes = result[0].boxes
        
        # Ensure result[0] exists and has the expected attributes
        if not hasattr(result[0], 'boxes') or not hasattr(result[0], 'names'):
            raise ValueError("Model response does not contain expected attributes 'boxes' or 'names'")

        # Initialize counts for all classes
        counts = {name: 0 for name in result[0].names.values()}
        
        # Initialize variables for kidney stone detection
        kidney_stone_detected = False
        max_confidence = 0.0
        prediction_text = "No kidney stones detected"
        total_detections = 0
        
        # Handle case when detections are found
        if boxes is not None and len(boxes) > 0:
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
                        
                        kidney_stone_detected = True
                        class_name = names[class_id]
                        prediction_text = f"Kidney stones detected - {class_name} (Count: {class_count})"
            
            # If multiple classes detected, show summary
            if total_detections > 0:
                detected_classes = [names[class_id] for class_id in names.keys() if counts[names[class_id]] > 0]
                if len(detected_classes) > 1:
                    prediction_text = f"Kidney stones detected - Multiple types found: {', '.join(detected_classes)}"
        
        # Force cleanup
        del result
        gc.collect()
        
        # Calculate processing time
        processing_time = f"{time.time() - start_time:.2f} seconds"
        
        print(f"Detection counts: {counts}")
        print(f"Max confidence: {max_confidence}")
        print(f"Processing time: {processing_time}")
        
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
        # Force cleanup on error
        gc.collect()
        raise e