# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# from ultralytics import YOLO
# import random
# import wandb

# # Initialize Weights and Biases
# wandb.init(project='boxCluster')

# # Load YOLOv8 model
# model = YOLO('models/18junv11.pt')  # Replace with the path to your YOLOv8 weights

# # Load the video
# video_path = "feeds/osotspa/DJI_0709_shortened.MP4"
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # Define the desired class ID
# desired_class_id = 0  # Replace this with the actual class ID you want to detect

# # Grid search for hyperparameters
# eps_values = np.linspace(500, 550, 1)  # Example range for eps
# min_samples_values = range(1,2)  # Example range for min_samples

# best_score = -1
# best_params = {'eps': None, 'min_samples': None}

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference with YOLOv8
#     results = model(frame)

#     # Extract bounding boxes and confidences
#     final_boxes = []
#     confidences = []

#     for result in results:
#         boxes = result.boxes  # Assuming 'boxes' contains the bounding box information

#         for box in boxes:
#             class_id = box.cls[0].item()
#             if class_id == desired_class_id:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = box.conf[0].item()
#                 final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
#                 confidences.append(conf)

#     final_boxes = np.array(final_boxes)
#     confidences = np.array(confidences)

#     if len(final_boxes) > 0:
#         centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
        
#         for eps in eps_values:
#             for min_samples in min_samples_values:
#                 clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
#                 labels = clustering.labels_

#                 if len(set(labels)) > 1:
#                     score = silhouette_score(centers, labels)

#                     # Log hyperparameters and metrics to W&B
#                     wandb.log({
#                         'eps': eps,
#                         'min_samples': min_samples,
#                         'silhouette_score': score
#                     })

#                     if score > best_score:
#                         best_score = score
#                         best_params = {'eps': eps, 'min_samples': min_samples}

# # Create the output video path with the best silhouette score
# output_path = f"output_video_{best_score:.2f}.mp4"
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Reset the video capture to the beginning
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference with YOLOv8
#     results = model(frame)

#     # Extract bounding boxes and confidences
#     final_boxes = []
#     confidences = []

#     for result in results:
#         boxes = result.boxes  # Assuming 'boxes' contains the bounding box information

#         for box in boxes:
#             class_id = box.cls[0].item()
#             if class_id == desired_class_id:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = box.conf[0].item()
#                 final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
#                 confidences.append(conf)

#     final_boxes = np.array(final_boxes)
#     confidences = np.array(confidences)

#     if len(final_boxes) > 0:
#         centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
#         clustering = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(centers)
#         labels = clustering.labels_

#         # Count the number of detections in each cluster
#         box_counts = {label: list(labels).count(label) for label in np.unique(labels)}

#         # Define a color map for clusters
#         colors = {label: [random.randint(0, 255) for _ in range(3)] for label in np.unique(labels)}

#         # Draw bounding boxes with different colors for each cluster and confidence scores
#         for i, box in enumerate(final_boxes):
#             x, y, w, h = box
#             label = labels[i]
#             color = colors[label]
#             conf = confidences[i]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
#             cv2.putText(frame, f'{conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2, cv2.LINE_AA)

#         # Draw a bounding box around each cluster with count
#         for label in np.unique(labels):
#             cluster_boxes = final_boxes[labels == label]
#             if len(cluster_boxes) > 0:
#                 x_min = min(cluster_boxes[:, 0])
#                 y_min = min(cluster_boxes[:, 1])
#                 x_max = max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
#                 y_max = max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[label], 3)
                
#                 text = f'boxes: {box_counts[label]}'
#                 font_scale = 1
#                 thickness = 2  # Increase thickness for bold effect
#                 color = colors[label]
                
#                 # Get the text size
#                 (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
#                 # Draw a filled rectangle behind the text
#                 cv2.rectangle(frame, (x_min, y_min - 30 - text_height), (x_min + text_width, y_min - 30), (0, 0, 0), cv2.FILLED)
                
#                 # Put the text on top of the rectangle
#                 cv2.putText(frame, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

#     # Write the frame to the output video
#     out.write(frame)

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # Log the best parameters to W&B
# wandb.config.update(best_params)
# wandb.log({'best_silhouette_score': best_score})

# # Finish the W&B run
# wandb.finish()


import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('models/20junv13.pt')  # Replace 'yolov8.pt' with the path to your YOLOv8 weights

# Load the video
video_path = "feeds/osotspa/DJI_0709.MP4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output_video_long_0.8.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define the desired class ID
desired_class_id = 0  # Replace this with the actual class ID you want to detect

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLOv8
    results = model(frame, conf=0.8)

    # Extract bounding boxes and confidences
    final_boxes = []
    confidences = []

    for result in results:
        boxes = result.boxes  # Assuming 'boxes' contains the bounding box information

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

        # Count the number of detections in each cluster
        box_counts = {label: list(labels).count(label) for label in np.unique(labels) if label != -1}

        color = [0, 255, 0]  # Lime green color
        colors = {label: color for label in np.unique(labels) if label != -1}

        # Draw bounding boxes with lime green color for each cluster and confidence scores
        for i, box in enumerate(final_boxes):
            x, y, w, h = box
            label = labels[i]
            if label != -1:
                conf = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, f'{conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

        # Draw a bounding box around each cluster with count
        for label in np.unique(labels):
            if label != -1:
                cluster_boxes = final_boxes[labels == label]
                if len(cluster_boxes) > 0:
                    x_min = min(cluster_boxes[:, 0])
                    y_min = min(cluster_boxes[:, 1])
                    x_max = max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
                    y_max = max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                    
                    text = f'boxes: {box_counts[label]}'
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
