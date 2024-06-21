# from ultralytics import YOLO

# model = YOLO("fri.pt")  # Load a custom trained model

# # Perform tracking with the model
# results = model("feeds/test3.mp4", show=True, tracker="bytetrack.yaml", classes=[0], conf=0.8, iou=0.1)  # with ByteTrack



import cv2
from ultralytics import YOLO

# Load a custom trained model
model = YOLO("models/18junv11.pt")

# Open video file
cap = cv2.VideoCapture("feeds/osotspa/DJI_0714.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame, tracker="bytetrack.yaml", conf=0.6, iou=0.1)
    
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

        # Draw bounding boxes on the frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int for OpenCV
            conf = box.conf[0]
            cls = box.cls[0]
            label = f'{model.names[int(cls)]}: {conf:.2f}'

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate text size
            font_scale = 1.0
            font_thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_w, text_h = text_size

            # Set the text background rectangle coordinates
            rectangle_bgr = (0, 0, 0)
            box_coords = ((x1, y1 - 10), (x1 + text_w, y1 - 10 - text_h - 5))
            cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

            # Draw the text on top of the rectangle
            cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        # Optionally, draw masks and probabilities if needed
        # Note: Mask drawing requires additional processing

    # Display the frame with OpenCV
    cv2.imshow("YOLO Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
