# # process a single frame and save the results to a text file

# import cv2
# import numpy as np
# from sklearn.cluster import HDBSCAN
# from ultralytics import YOLO
# import re

# model = YOLO('models/20junv13.pt') 
# frame_path = "datasets/set2/4549_7043.png"  # Path to the frame you want to process

# # Read the frame
# frame = cv2.imread(frame_path)

# # Perform inference with YOLO
# results = model(frame)

# # Extract bounding boxes and confidences
# final_boxes = []
# confidences = []
# positions = []

# for result in results:
#     boxes = result.boxes 

#     for box in boxes:
#         class_id = box.cls[0].item()
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         conf = box.conf[0].item()
#         final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
#         confidences.append(conf)
#         positions.append((x1, y1))

# final_boxes = np.array(final_boxes)
# confidences = np.array(confidences)
# positions = np.array(positions)

# # Perform clustering (HDBSCAN)
# if len(final_boxes) > 0:
#     centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in final_boxes])
#     clustering = HDBSCAN(min_cluster_size=9, min_samples=3).fit(centers)
#     labels = clustering.labels_

#     # Prepare dictionary to store results
#     results_dict = {
#         'clusterBoundingBoxesPosition': [],
#         'numbeOfBoxesFoundInEachCluster': [],
#         'positionOfBoxesInEachCluster': [],
#         'confidenceOfBoxesInEachCluster': []
#     }

#     # Iterate over clusters
#     for label in np.unique(labels):
#         if label != -1:
#             cluster_boxes = final_boxes[labels == label]
#             cluster_positions = positions[labels == label]
#             cluster_confidences = confidences[labels == label]

#             # Store cluster information
#             results_dict['clusterBoundingBoxesPosition'].append(cluster_boxes.tolist())
#             results_dict['numbeOfBoxesFoundInEachCluster'].append(len(cluster_boxes))
#             results_dict['positionOfBoxesInEachCluster'].append(cluster_positions.tolist())
#             results_dict['confidenceOfBoxesInEachCluster'].append(cluster_confidences.tolist())

#     # Output results to a text file as a Python dictionary
#     output_file = 'results/frame_results.txt'
#     with open(output_file, 'w') as f:
#         f.write(str(results_dict))

#     print(f"Results saved to {output_file}")
# else:
#     print("No objects detected.")




import cv2
import numpy as np
from sklearn.cluster import HDBSCAN
from ultralytics import YOLO
import re

model = YOLO('models/20junv13.pt') 
frame_path = "datasets/set2/4549_7043.png"  



# Read the frame
frame = cv2.imread(frame_path)

# Perform inference with YOLO
results = model(frame)

# Extract bounding boxes and confidences
final_boxes = []
confidences = []
positions = []

for result in results:
    boxes = result.boxes 

    for box in boxes:
        class_id = box.cls[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        final_boxes.append([x1, y1, x2, y2])
        confidences.append(conf)
        positions.append(((x1 + x2) // 2, (y1 + y2) // 2))  # Calculate center of the box

final_boxes = np.array(final_boxes)
confidences = np.array(confidences)
positions = np.array(positions)

# Perform clustering (HDBSCAN)
if len(final_boxes) > 0:
    clustering = HDBSCAN(min_cluster_size=9, min_samples=3).fit(positions)
    labels = clustering.labels_

    # Prepare dictionary to store results
    results_dict = {
        'clusterBoundingBoxesPosition': [],
        'numbeOfBoxesFoundInEachCluster': [],
        'positionOfBoxesInEachCluster': [],
        'confidenceOfBoxesInEachCluster': []
    }

    # Iterate over clusters
    for label in np.unique(labels):
        if label != -1:
            cluster_boxes = final_boxes[labels == label]
            cluster_positions = positions[labels == label]
            cluster_confidences = confidences[labels == label]

            # Calculate the bounding box that encapsulates all boxes in the cluster
            x_min = np.min(cluster_boxes[:, 0])
            y_min = np.min(cluster_boxes[:, 1])
            x_max = np.max(cluster_boxes[:, 2])
            y_max = np.max(cluster_boxes[:, 3])

            cluster_bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]

            # Store cluster information
            results_dict['clusterBoundingBoxesPosition'].append(cluster_bounding_box)
            results_dict['numbeOfBoxesFoundInEachCluster'].append(len(cluster_boxes))
            results_dict['positionOfBoxesInEachCluster'].append(cluster_positions.tolist())
            results_dict['confidenceOfBoxesInEachCluster'].append(cluster_confidences.tolist())

    # Output results to a text file as a Python dictionary
    output_file = 'results/frame_results.txt'
    with open(output_file, 'w') as f:
        f.write(str(results_dict))

    print(f"Results saved to {output_file}")
else:
    print("No objects detected.")