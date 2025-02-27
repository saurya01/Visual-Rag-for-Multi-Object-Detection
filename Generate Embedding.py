import torch
import faiss
import open_clip
import numpy as np
import cv2
import os
import json
import yaml
import logging
from PIL import Image

# Set up logging
logging.basicConfig(
    filename="indexing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load YAML Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config file
images_path = config["dataset"]["images_path"]
labels_path = config["dataset"]["labels_path"]
faiss_index_path = config["faiss"]["index_path"]
metadata_path = config["faiss"]["metadata_path"]
class_labels = config["labels"]["names"]

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
model.to(device)
logging.info(f"Using device: {device}")

# FAISS index setup
embedding_dim = 512  # CLIP ViT-B/32 output size
index = faiss.IndexFlatL2(embedding_dim)
image_store = {}

def load_yolo_annotations(image_filename):
    """Load YOLO format annotations and return bounding boxes + labels."""
    label_filename = os.path.join(labels_path, os.path.splitext(image_filename)[0] + ".txt")
    
    if not os.path.exists(label_filename):
        logging.warning(f"Annotation file not found: {label_filename}")
        return []
    
    with open(label_filename, "r") as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            logging.warning(f"Invalid annotation format in {label_filename}: {line}")
            continue
        
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        objects.append((class_id, x_center, y_center, width, height))
    
    return objects

def crop_object(image, bbox):
    """Crop object from image using YOLO bounding box format."""
    h, w, _ = image.shape
    class_id, x_center, y_center, width, height = bbox

    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    cropped = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return cropped, class_id

def compute_embedding(image):
    """Compute CLIP embedding for an object crop."""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).cpu().numpy()
    
    return embedding

def process_dataset():
    """Process dataset: extract object crops, compute embeddings, and store in FAISS."""
    obj_count = 0
    for image_filename in os.listdir(images_path):
        print(obj_count)
        image_path = os.path.join(images_path, image_filename)
        if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            continue

        objects = load_yolo_annotations(image_filename)

        for bbox in objects:
            cropped, class_id = crop_object(image, bbox)
            if cropped.size == 0:
                logging.warning(f"Skipping empty crop in {image_filename}")
                continue
            
            embedding = compute_embedding(cropped)
            index.add(embedding)  
            
            image_store[obj_count] = {
                "image_path": image_path,
                "class_id": class_id,
                "class_name": class_labels[class_id] if class_id < len(class_labels) else "unknown"
            }
            obj_count += 1
            logging.info(f"Processed object {obj_count}: {image_path} - {class_labels[class_id]}")
    
    logging.info(f"Total objects processed: {obj_count}")

def save_faiss_index():
    """Save FAISS index and metadata."""
    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, "w") as f:
        json.dump(image_store, f)

    logging.info(f"FAISS index and metadata saved. Objects indexed: {len(image_store)}")

# Run processing
logging.info("Starting dataset indexing...")
process_dataset()
save_faiss_index()
logging.info("Indexing completed successfully.")
print(f"[INFO] FAISS index and metadata saved successfully with {len(image_store)} objects.")
