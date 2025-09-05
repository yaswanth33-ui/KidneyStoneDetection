from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import os
import sys
import io
from base64 import b64encode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'artifacts', 'best.pt')
        model_exists = os.path.exists(model_path)
        return jsonify({
            'status': 'healthy',
            'model_path': model_path,
            'model_exists': model_exists,
            'server_running': True,
            'working_directory': os.getcwd(),
            'script_directory': script_dir
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test-upload', methods=['POST'])
def test_upload():
    """Test endpoint that processes images without ML model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected'}), 400
            
        # Just process the image without ML
        image = Image.open(file.stream)
        
        # Convert to base64 for return
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        img_str = b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'result_image': img_str,
            'prediction': 'Test mode - no ML processing',
            'confidence': 0.95,
            'processing_time': '0.1 seconds',
            'image_dimensions': f"{image.size[0]}x{image.size[1]}",
            'kidney_stones_detected': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        print(f"Received file: {file.filename}")
        
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}
        file_extension = os.path.splitext(file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            print(f"Unsupported file format: {file_extension}")
            return jsonify({'error': f'Unsupported file format: {file_extension}. Supported formats: {", ".join(allowed_extensions)}'}), 400
            
        try:
            image = Image.open(file.stream)
            print(f"Image opened successfully: {image.size}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted image to RGB mode")
            result_data = predict(image)
            print(f"Prediction completed. Results: {result_data['prediction']}")
            
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
    # Use environment variables for production settings
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    port = int(os.getenv('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)