import cv2
import torch
import numpy as np
import argparse
import time
import yaml
import os
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.torch_utils import select_device

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input video")
ap.add_argument("-o", "--output", required=True, help="Path to output video")
ap.add_argument("-w", "--weights", required=True, help="Path to YOLOv7 weights")
ap.add_argument("-c", "--confidence", type=float, default=0.25, help="Confidence threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.45, help="NMS threshold")
ap.add_argument("-n", "--names", required=True, help="Path to .names file")
args = vars(ap.parse_args())




# Load YAML file
with open(args["names"], "r") as f:
    data = yaml.safe_load(f)
LABELS = data['names']



# Colors for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLOv7 model
print("[INFO] Loading YOLOv7 model...")
device = select_device('0')  # Use GPU (0) or CPU ('cpu')
model = attempt_load(args["weights"], map_location=device)
model.eval()

# Video capture and output
video_path = args["input"]
output_path = args["output"]
cap = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)

# Performance metrics
total_frames = 0
total_detections = 0
class_counts = {label: 0 for label in LABELS}
inference_times = []

# Process each frame
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Preprocess frame for YOLOv7
    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, args["confidence"], args["threshold"])

    # Draw detections
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{LABELS[int(cls)]} {conf:.2f}"
                color = [int(c) for c in COLORS[int(cls)]]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                total_detections += 1
                class_counts[LABELS[int(cls)]] += 1

    # Calculate inference time
    end_time = time.time()
    inference_times.append(end_time - start_time)

    # Write output
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)
    writer.write(frame)

    # Optional: Display the frame
    cv2.imshow("Cherry Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
writer.release()
cv2.destroyAllWindows()

# Calculate performance metrics
average_inference_time = sum(inference_times) / len(inference_times)
fps = 1 / average_inference_time

# Print results
print("[INFO] Loaded Class Names:", LABELS)
print("\n[INFO] Performance Metrics:")
print(f"Total Frames Processed: {total_frames}")
print(f"Total Detections: {total_detections}")
print(f"Average Inference Time per Frame: {average_inference_time:.4f} seconds")
print(f"FPS (Frames Per Second): {fps:.2f}")
print("Detections per Class:")
for label, count in class_counts.items():
    print(f"{label}: {count}")
