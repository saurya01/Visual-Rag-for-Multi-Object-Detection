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

def draw_bounding_boxes(image, boxes, labels, scores):
    """Draw bounding boxes with labels and confidence scores."""
    for (box, label, score) in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # Green color for boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {score:.2f}"  # Direct confidence score
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def process_results(results, boxes):
    """Filter objects dynamically based on FAISS distance threshold."""
    """Dynamically filter objects based on FAISS distance distribution."""
    distances = np.array([res["distance"] for res in results])
    
    if len(distances) == 0:
        return [], []

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + std_dist  # Keep only results within 1 std deviation
    
    filtered_results = [(res, box) for res, box in zip(results, boxes) if res["distance"] <= threshold]

    final_results = [res for res, _ in filtered_results]
    final_boxes = [box for _, box in filtered_results]

    return final_results, final_boxes

def main(image_path, top_k):
    """Process query image and return top-k labels with matched images."""
    print(f"[INFO] Processing image: {image_path}")

    # Detect objects
    object_crops, boxes, query_image = detect_objects(image_path)
    if not object_crops:
        print("[WARNING] No objects detected in the image.")
        return

    results = []
    for i, crop in enumerate(object_crops):
        embedding = compute_embedding(crop)
        faiss_results = search_faiss(embedding, top_k)
        if not faiss_results:
            print(f"[WARNING] No FAISS match found for object {i+1}.")
            continue

        match = faiss_results[0]  # Take top-1 result
        confidence = 100 - match["distance"]  # Convert FAISS distance to confidence
        match["confidence"] = confidence  # Store confidence
        results.append(match)

        print(f"\n[Object {i+1}] Predicted Category: {match['class_name']} (Confidence: {confidence:.2f})")

    # Filter results dynamically
    filtered_results, filtered_boxes = process_results(results, boxes)

    if not filtered_results:
        print("[INFO] No objects met the confidence threshold.")
        return

    # Show matched images with bounding boxes
    query_image = draw_bounding_boxes(query_image, filtered_boxes, 
                                      [res["class_name"] for res in filtered_results],
                                      [res["confidence"] for res in filtered_results])
    
    # Display only query image with relevant bounding boxes
    cv2.imshow("Query Image (Filtered Detections)", query_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to query image")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top results to return")
    args = parser.parse_args()

    main(args.image_path, args.top_k)
