
import os
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("18junv10.pt")

# Define path to the image file
source = "./datasets/set1"

# Run inference on the source
results = model(source, conf=0.6)  # list of Results objects


save_dir = './datasets/set1_results'
os.makedirs(save_dir, exist_ok=True)

# Iterate over each result object
for i, result in enumerate(results):
    boxes = result.boxes  # Assuming these are necessary for the file name
    masks = result.masks  # Assuming these are necessary for the file name
    keypoints = result.keypoints  # Assuming these are necessary for the file name
    probs = result.probs  # Assuming these are necessary for the file name
    obb = result.obb  # Assuming these are necessary for the file name
    
    # Generate a specific filename for each result
    filename = f"result_{i}.jpg"  # Example: result_0.jpg, result_1.jpg, etc.
    
    # Construct the full file path
    filepath = os.path.join(save_dir, filename)
    
    try:
        # Save the result to disk
        result.save(filename=filepath)
        print(f"Saved result {i} to: {filepath}")
    except Exception as e:
        print(f"Failed to save result {i}: {str(e)}")
