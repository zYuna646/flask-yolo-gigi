from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)  # Allow all origins, or use `CORS(app, origins=["your-origin"])` to specify origins

# Load your custom YOLOv8 model from the .pt file
MODEL_PATH = './model.pt'  # Adjust the path to your .pt file
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process the image
    processed_image_path = process_image(filepath)

    # Return the processed image
    return send_file(processed_image_path, mimetype='image/jpeg')

def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Run inference on the image using YOLOv8 model
    results = model(img)

    # Get bounding boxes and labels
    detections = results[0].boxes  # Get detection boxes
    names = model.names  # Get class names

    # Loop through the detections
    for detection in detections:
        # Get box coordinates, confidence, and class
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # Convert coordinates to integers
        conf = detection.conf[0].item()  # Confidence score
        cls = int(detection.cls[0].item())  # Class index

        # Draw a bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

        # Get label and draw it
        label = f"{names[cls]} {conf:.2f}"  # Class name and confidence
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the processed image with annotations
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, img)  # Save the annotated image

    return processed_image_path

if __name__ == '__main__':
    app.run(debug=True)
