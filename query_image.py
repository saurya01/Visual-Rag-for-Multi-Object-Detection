import torch
import faiss
import open_clip
import numpy as np
import cv2
import json
import yaml
import argparse
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import google.auth.credentials
import google.auth.transport.requests
from google.generativeai import GenerativeModel, configure

# Load configuration from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
faiss_index_path = config["faiss"]["index_path"]
metadata_path = config["faiss"]["metadata_path"]
class_labels = config["labels"]["names"]

# Load FAISS index and metadata
index = faiss.read_index(faiss_index_path)
with open(metadata_path, "r") as f:
    image_store = json.load(f)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
model.to(device)

# Load YOLOv8 model for object detection
yolo_model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" or your fine-tuned model

# Initialize Gemini model
use_gemini = True
try:
    # Replace this with your actual API key
    GOOGLE_API_KEY = "AIzaSyDXEOhpsO-VBF4_oWrNuX27vJ0rbm0yqeU"
    
    # Configure the Gemini API with the key
    configure(api_key=GOOGLE_API_KEY)
    
    # Initialize Gemini model
    gemini = GenerativeModel("gemini-1.5-flash")
    print("[INFO] Gemini API initialized successfully with embedded API key.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Gemini: {e}")
    use_gemini = False

def detect_objects(image_path):
    """Run YOLOv8 on the image and return detected object crops & bounding boxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    results = yolo_model(image)
    crops = []
    boxes = []

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            crops.append(cropped)
            boxes.append((x1, y1, x2, y2))

    return crops, boxes, image

def compute_embedding(image):
    """Compute CLIP embedding for an object crop."""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).cpu().numpy()
    
    return embedding

def search_faiss(query_embedding, top_k=3):
    """Search FAISS for the closest matches and return metadata."""
    distances, indices = index.search(query_embedding, top_k)
    results = []

    for i, idx in enumerate(indices[0]):
        if str(idx) in image_store:
            result = image_store[str(idx)]
            result["distance"] = float(distances[0][i])  # Store confidence score
            results.append(result)

    return results

def prepare_image_for_gemini(cv_image):
    """Convert OpenCV image to format suitable for Gemini API."""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    # Convert to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    # Return image bytes
    return image_bytes

def verify_with_gemini(query_image, retrieved_image):
    """Use Google Gemini to verify if both images match."""
    try:
        # Convert OpenCV images to bytes format for Gemini
        query_bytes = prepare_image_for_gemini(query_image)
        retrieved_bytes = prepare_image_for_gemini(retrieved_image)
        
        # Format properly for Gemini API according to error message
        query_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(query_bytes).decode('utf-8')
        }
        
        retrieved_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(retrieved_bytes).decode('utf-8')
        }
        
        # Create content object with parts
        content = {
            "parts": [
                {"inline_data": query_part},
                {"inline_data": retrieved_part},
                {"text": "Are these two images of the same object?"}
            ]
        }
        
        response = gemini.generate_content(content)
        
        return "yes" in response.text.lower()
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        # Default to False on failure
        return False

def draw_bounding_boxes(image, boxes, labels, scores):
    """Draw bounding boxes with labels and confidence scores."""
    for (box, label, score) in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # Green color for boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {score:.2f}"  # Direct confidence score
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def main(image_path, top_k):
    """Process query image and return top-k labels with matched images."""
    print(f"[INFO] Processing image: {image_path}")

    # Detect objects
    object_crops, boxes, query_image = detect_objects(image_path)
    if not object_crops:
        print("[WARNING] No objects detected in the image.")
        return

    verified_results = []
    verified_boxes = []
    
    for i, (crop, box) in enumerate(zip(object_crops, boxes)):
        print(f"[INFO] Processing object {i+1}")
        embedding = compute_embedding(crop)
        faiss_results = search_faiss(embedding, top_k)
        
        if not faiss_results:
            print(f"[WARNING] No FAISS match found for object {i+1}.")
            continue
        
        object_verified = False
        
        for match in faiss_results:
            retrieved_image_path = match["image_path"]
            try:
                retrieved_image = cv2.imread(retrieved_image_path)
                if retrieved_image is None:
                    print(f"[ERROR] Failed to load retrieved image: {retrieved_image_path}")
                    continue
                
                # Verify using Gemini
                if verify_with_gemini(crop, retrieved_image):
                    # Convert distance to confidence score (lower distance = higher confidence)
                    match["confidence"] = max(0, min(100, 100 * (1 - match["distance"] / 2)))
                    match["verification_method"] = "Gemini"
                    verified_results.append(match)
                    verified_boxes.append(box)
                    object_verified = True
                    print(f"[Object {i+1}] Verified Category: {match['class_name']} (by Gemini)")
                    break  # Stop after first verification for this object
                else:
                    print(f"[Object {i+1}] Match rejected by Gemini verification.")
            except Exception as e:
                print(f"[ERROR] Error processing match: {e}")
        
        if not object_verified:
            print(f"[INFO] No verified matches for object {i+1}")

    if not verified_results:
        print("[INFO] No objects met verification criteria.")
        return

    # Draw bounding boxes on verified results
    query_image = draw_bounding_boxes(
        query_image, 
        verified_boxes,
        [f"{res['class_name']} (Gemini)" for res in verified_results],
        [res["confidence"] for res in verified_results]
    )
    
    # Save result image
    result_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(result_path, query_image)
    print(f"[INFO] Result saved to {result_path}")
    
    # Display result
    cv2.imshow("Query Image (Verified Detections)", query_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to query image")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top results to return")
    args = parser.parse_args()

    main(args.image_path, args.top_k)
