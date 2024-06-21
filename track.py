from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolov8n.pt")  # Load an official Detect model
# model = YOLO("yolov8n-seg.pt")  # Load an official Segment model
# model = YOLO("yolov8n-pose.pt")  # Load an official Pose model
model = YOLO("models/18junv10.pt")  # Load a custom trained model

# Perform tracking with the model
results = model.track("feeds/processed/output.mp4", show=True, classes=[0], tracker="bytetrack.yaml")  # with ByteTrack
# results = model.track("feeds/test.mp4", show=True, classes=[0], tracker="bytetrack.yaml")  # with ByteTrack
