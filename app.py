from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import cv2
import base64
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app, resources={r"/*": {"origins": "*"}})  # This allows requests from any origin

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
    processed_image_path, unique_labels = process_image(filepath)

    # Convert the processed image to base64
    with open(processed_image_path, "rb") as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create a response that includes the base64 image data and unique labels
    response = {
        "labels": unique_labels,
        "image_data": base64_encoded_image  # Base64 encoded image
    }

    # Return the processed image data along with label list as JSON response
    return jsonify(response), 200

def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Run inference on the image using YOLOv8 model
    results = model(img)

    # Get bounding boxes and labels
    detections = results[0].boxes  # Get detection boxes
    names = model.names  # Get class names
    unique_labels = set()  # Set to store unique labels

    # Loop through the detections
    for detection in detections:
        # Get box coordinates, confidence, and class
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # Convert coordinates to integers
        conf = detection.conf[0].item()  # Confidence score
        cls = int(detection.cls[0].item())  # Class index

        # Add class name to the unique labels set
        unique_labels.add(names[cls])

        # Draw a bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

        # Get label and draw it
        label = f"{names[cls]} {conf:.2f}"  # Class name and confidence
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the processed image with annotations
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, img)  # Save the annotated image

    return processed_image_path, list(unique_labels)  # Return path and unique labels as list

if __name__ == '__main__':
    app.run(debug=True)
