# top layer only
import torch
import cv2
import numpy as np
from sklearn.cluster import HDBSCAN
from ultralytics import YOLO


# model = YOLO('models/18junv11.pt') 
model = YOLO('models/20junv13.pt') 
video_path = "feeds/osotspa/DJI_0708_shortened.MP4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "onlytop.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

desired_class_id = 0

# top layer y range
top_layer_tolerance = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLOv8
    results = model(frame)

    # Extract bounding boxes and confidences
    final_boxes = []
    confidences = []

    for result in results:
        boxes = result.boxes 

        for box in boxes:
            class_id = box.cls[0].item()
            if class_id == desired_class_id:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(conf)

    final_boxes = np.array(final_boxes)
    confidences = np.array(confidences)

    if len(final_boxes) > 0:
        # Cluster boxes by pallet using HDBSCAN
        centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
        clustering = HDBSCAN(min_cluster_size=9, min_samples=3).fit(centers)
        labels = clustering.labels_

        # Define colors
        color = [0, 255, 0]  
        top_layer_color = [255, 105, 180]

        # Identify the top-most layer boxes within each cluster
        top_layer_boxes = []
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                if len(cluster_boxes) > 0:
                    # Identify the top-most layer (smallest y coordinate) within the cluster
                    cluster_top_y = min(cluster_boxes[:, 1])
                    top_layer_boxes.extend([
                        box for box in cluster_boxes
                        if cluster_top_y - top_layer_tolerance <= box[1] <= cluster_top_y + top_layer_tolerance
                    ])

        top_layer_boxes = np.array(top_layer_boxes)  # Convert to numpy array for element-wise comparison

        # Draw bounding boxes with different colors for top layer and other boxes
        for i, box in enumerate(final_boxes):
            x, y, w, h = box
            label = labels[i]
            conf = confidences[i]
            if label != -1:
                box_color = top_layer_color if any((box == top_box).all() for top_box in top_layer_boxes) else color
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 4)
                cv2.putText(frame, f'{conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

        # Draw a bounding box around each cluster with count of top layer boxes
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                if len(cluster_boxes) > 0:
                    x_min = min(cluster_boxes[:, 0])
                    y_min = min(cluster_boxes[:, 1])
                    x_max = max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
                    y_max = max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Count top layer boxes for the cluster
                    top_layer_count = sum(
                        1 for box in cluster_boxes
                        if any((box == top_box).all() for top_box in top_layer_boxes)
                    )

                    # Draw the count of top layer boxes for the cluster
                    text = f'boxes: {top_layer_count}'
                    font_scale = 2
                    thickness = 2  # Increase thickness for bold effect

                    # Get the text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Draw a filled rectangle behind the text
                    cv2.rectangle(frame, (x_min, y_min - 30 - text_height), (x_min + text_width, y_min - 30), (0, 0, 0), cv2.FILLED)

                    # Put the text on top of the rectangle
                    cv2.putText(frame, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
