import cv2
import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
from ultralytics import YOLO
import re

model = YOLO('models/20junv13.pt') 
video_path = "feeds/osotspa/DJI_0707_shortened.MP4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
pattern = r'/([^/]+)\.MP4$'
match = re.search(pattern, video_path)

if match:
    filename = match.group(1)
    output_path = f"results/{filename}.mp4"

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

front_class_id = 0
top_class_id = 1

top_layer_tolerance = 50  

def generate_colors(num_colors):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3)).tolist()
    return colors

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLO
    results = model(frame)

    # Extract bounding boxes and confidences
    final_boxes = []
    confidences = []
    class_ids = []

    for result in results:
        boxes = result.boxes 

        for box in boxes:
            class_id = box.cls[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(conf)
            class_ids.append(class_id)

    final_boxes = np.array(final_boxes)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    if len(final_boxes) > 0:
        centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
        clustering = HDBSCAN(min_cluster_size=9, min_samples=3).fit(centers)
        # clustering = DBSCAN(eps=200, min_samples=1).fit(centers)
        labels = clustering.labels_

        class_1_color = [255, 0, 0]       
        cluster_bbox_color = [0, 255, 0] 

        # Identify the number of rows for class 0 within each cluster and assign colors
        rows_per_cluster = {}
        row_colors = {}
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                cluster_class_ids = class_ids[labels == label]
                if len(cluster_boxes) > 0:
                    # Filter class 0 boxes in the cluster
                    class_0_boxes = cluster_boxes[cluster_class_ids == front_class_id]
                    if len(class_0_boxes) > 0:
                        # Identify the number of rows by looking at the unique y coordinates within tolerance
                        unique_y_coords = []
                        for box in class_0_boxes:
                            y_coord = box[1]
                            if not any(abs(y - y_coord) <= top_layer_tolerance for y in unique_y_coords):
                                unique_y_coords.append(y_coord)
                        num_rows = len(unique_y_coords)
                        rows_per_cluster[label] = num_rows
                        row_colors[label] = generate_colors(num_rows)

        # Draw bounding boxes for each row of class 0 boxes using different colors
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                cluster_class_ids = class_ids[labels == label]
                if len(cluster_boxes) > 0:
                    class_0_boxes = cluster_boxes[cluster_class_ids == front_class_id]
                    if len(class_0_boxes) > 0:
                        unique_y_coords = []
                        row_color_map = {}
                        for box in class_0_boxes:
                            y_coord = box[1]
                            if not any(abs(y - y_coord) <= top_layer_tolerance for y in unique_y_coords):
                                unique_y_coords.append(y_coord)
                            row_index = unique_y_coords.index(min(unique_y_coords, key=lambda y: abs(y - y_coord)))
                            row_color_map[y_coord] = row_colors[label][row_index]
                        
                        for box in class_0_boxes:
                            x, y, w, h = box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), row_color_map[box[1]], 4)

        # Draw bounding boxes for class 1 boxes
        for i, box in enumerate(final_boxes):
            if class_ids[i] == top_class_id:  # <-- Correct comparison
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), class_1_color, 4)
                
        # Draw a bounding box around each cluster with count of top layer boxes multiplied by number of "1" class detections
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                cluster_class_ids = class_ids[labels == label]
                if len(cluster_boxes) > 0:
                    # Cluster bounding box
                    x_min = min(cluster_boxes[:, 0])
                    y_min = min(cluster_boxes[:, 1])
                    x_max = max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
                    y_max = max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), cluster_bbox_color, 2)

                    # Count the number of rows in the cluster
                    num_rows = rows_per_cluster.get(label, 0)

                    # Count the number of "1" class detections in the cluster
                    class_1_count = sum(cluster_class_ids == top_class_id)

                    # Count the number of class 0 detections in the cluster
                    class_0_count = sum(cluster_class_ids == front_class_id)

                    # Multiply the number of rows by the number of "1" class detections
                    total_count = num_rows * top_class_id

                    # Draw the count of rows multiplied by class 1 detections for the cluster
                    text = f'boxes = {total_count}, rows = {num_rows}, top = {class_1_count}, front = {class_0_count}'
                    font_scale = 1.2
                    thickness = 2

                    # Get the text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Draw a filled rectangle behind the text
                    cv2.rectangle(frame, (x_min, y_min - 30 - text_height), (x_min + text_width, y_min - 30), (0, 0, 0), cv2.FILLED)

                    # Put the text on top of the rectangle
                    cv2.putText(frame, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, cluster_bbox_color, thickness, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
