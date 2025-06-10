from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import os
from utils import predict

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('home.html', latest_analysis=None)

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        model_path = os.path.join(os.getcwd(), 'artifacts', 'best.pt')
        model_exists = os.path.exists(model_path)
        return jsonify({
            'status': 'healthy',
            'model_path': model_path,
            'model_exists': model_exists,
            'server_running': True
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '' or file.filename is None:
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        print(f"Received file: {file.filename}")
        
        # Check file format
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}
        file_extension = os.path.splitext(file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            print(f"Unsupported file format: {file_extension}")
            return jsonify({'error': f'Unsupported file format: {file_extension}. Supported formats: {", ".join(allowed_extensions)}'}), 400
            
        try:
            # Open and validate image
            image = Image.open(file.stream)
            print(f"Image opened successfully: {image.size}")
              # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted image to RGB mode")
                
            # Call predict function and get results
            result_data = predict(image)
            print(f"Prediction completed. Results: {result_data['prediction']}")
            
            # Return JSON response with the results
            return jsonify({
                'success': True,
                'result_image': result_data['result_image'],
                'prediction': result_data['prediction'],
                'confidence': result_data['confidence'],
                'counts': result_data['counts'],
                'processing_time': result_data['processing_time'],
                'image_dimensions': result_data['image_dimensions'],
                'kidney_stones_detected': result_data['kidney_stones_detected']
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
            
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)