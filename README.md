# Kidney Stone Detection

## Overview

This project is a Flask-based web application designed to detect kidney stones from uploaded medical images. The application uses a YOLO-v8s deep learning model to analyze the images and provide predictions about the presence of kidney stones.

## Features

- **Health Check Endpoint**: Verify the server's health and model availability.
- **Image Upload**: Upload medical images for kidney stone detection.
- **Prediction Results**: Get detailed results including confidence levels, counts, and processing time.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yaswanth33-ui/KidneyStoneDetection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd KidneyStoneDetection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python server/app.py
   ```

## Usage

1. Open a web browser and navigate to `http://localhost:5000`.
2. Use the upload feature to submit medical images for analysis.
3. View the prediction results on the web interface.

## Project Structure

- `server/`: Contains the Flask application code.
- `model/`: Includes the deep learning models and related files.
- `dataset/`: Contains training, validation, and test datasets.
- `artifacts/`: Stores the trained model files.
- `templates/`: HTML templates for the web interface.

## Model Details

The application uses the YOLO-v8s deep learning model for kidney stone detection. Below are the specifics:

- **Model Architecture**: YOLO-v8s (You Only Look Once)
- **Training Dataset**: Medical images annotated for kidney stone detection.
- **Input Format**: RGB images in formats such as `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`, and `.tiff`.
- **Output**: Predictions include:
  - Detected kidney stones
  - Confidence levels
  - Counts
  - Processing time
  - Image dimensions

The trained model file is located in the `artifacts/` directory as `best.pt`. This file is loaded during runtime for predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact [yaswanthreddypanem@gmail.com].
