import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from ultralytics import YOLO
import random

model = YOLO('models/18junv11.pt') 

# Load the image
image_path = "datasets/set2/4549_286.png"
image = cv2.imread(image_path)
height, width, _ = image.shape

results = model(image)
# Extract bounding boxes and confidences
final_boxes = []
confidences = []

# Iterate through the results to extract bounding boxes and confidences
for result in results:
    boxes = result.boxes  # Assuming 'boxes' contains the bounding box information

    # Iterate through each box
    for box in boxes:
        # Convert the box to the format [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        # Append to the list
        final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        confidences.append(conf)

# Convert to numpy array
final_boxes = np.array(final_boxes)
confidences = np.array(confidences)

# Cluster boxes by pallet using DBSCAN
centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
clustering = DBSCAN(eps=200, min_samples=1).fit(centers)
labels = clustering.labels_

# Count the total number of detections
total_detections = len(final_boxes)

# Count the number of clusters (pallets)
num_clusters = len(np.unique(labels))

# Count the number of detections in each cluster
box_counts = {label: list(labels).count(label) for label in np.unique(labels)}

# Output the total number of detections and number of clusters
print(f"Total number of detections: {total_detections}")
print(f"Number of clusters (pallets): {num_clusters}")
for label in np.unique(labels):
    print(f"Cluster {label}: {box_counts[label]} detections")

# Define a color map for clusters
colors = {label: [random.randint(0, 255) for _ in range(3)] for label in np.unique(labels)}

# Draw bounding boxes with different colors for each cluster and confidence scores
for i, box in enumerate(final_boxes):
    x, y, w, h = box
    label = labels[i]
    color = colors[label]
    conf = confidences[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Add confidence score text
    cv2.putText(image, f'{conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

# Draw a bounding box around each cluster with count
for label in np.unique(labels):
    cluster_boxes = final_boxes[labels == label]
    if len(cluster_boxes) > 0:
        x_min = min(cluster_boxes[:, 0])
        y_min = min(cluster_boxes[:, 1])
        x_max = max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
        y_max = max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[label], 2)
        
        # Add text with the count of detections in the cluster
        text = f'boxes: {box_counts[label]}'
        font_scale = 1
        thickness = 2  # Increase thickness for bold effect
        color = colors[label]
        
        # Get the text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw a filled rectangle behind the text
        cv2.rectangle(image, (x_min, y_min - 30 - text_height), (x_min + text_width, y_min - 30), (0, 0, 0), cv2.FILLED)
        
        # Put the text on top of the rectangle
        cv2.putText(image, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# Save or display the image with bounding boxes, counts, and confidence scores
output_image_path = "detected_clusters.png"
cv2.imwrite(output_image_path, image)
