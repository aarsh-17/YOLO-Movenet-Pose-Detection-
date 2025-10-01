# ======================================
# Install dependencies if not installed
# ======================================
# pip install ultralytics opencv-python tensorflow numpy

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ======================================
# Load YOLOv8 model (pre-trained or fine-tuned)
# ======================================
model = YOLO("yolov8n.pt")  # or your custom .pt file

# ======================================
# Load MoveNet (Pose Estimation - TFLite)
# ======================================
interpreter = tf.lite.Interpreter(model_path="4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_idx = input_details[0]["index"]
output_idx = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
input_size = input_details[0]["shape"][1]  # usually 192 or 256

print(f"✅ MoveNet expects {input_dtype}, input size {input_size}x{input_size}")

# Pose estimation helper function
def run_movenet(img):
    img_resized = cv2.resize(img, (input_size, input_size))
    if input_dtype == np.float32:
        img_input = img_resized.astype(np.float32) / 255.0
    else:
        img_input = img_resized.astype(np.uint8)

    img_input = np.expand_dims(img_input, axis=0)
    interpreter.set_tensor(input_idx, img_input)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_idx)[0][0]  # (17, 3)
    return keypoints

# ======================================
# Open Webcam
# ======================================
cap = cv2.VideoCapture(0)  # 0 = default camera
cv2.namedWindow("YOLOv8n + MoveNet Pose", cv2.WINDOW_NORMAL)

# Skeleton pairs (keypoint connections)
SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),     # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),         # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_center = (w / 2, h / 2)

    # Detect persons with YOLO
    results = model(frame, conf=0.5, verbose=False)

    # Select the "best" person: largest + closest to center
    best_box = None
    best_score = -float("inf")

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Compute area and center distance
        box_w, box_h = x2 - x1, y2 - y1
        area = box_w * box_h

        cx, cy = x1 + box_w / 2, y1 + box_h / 2
        dist = ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5

        # Weighted scoring function (tune 0.5 weight if needed)
        score = area - 0.5 * dist

        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    # Run MoveNet only on the best person
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size > 0:
            keypoints = run_movenet(person_crop)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw keypoints
            kp_coords = []
            for ky, kx, kp_conf in keypoints:
                if kp_conf > 0.3:
                    cx = int(x1 + kx * (x2 - x1))
                    cy = int(y1 + ky * (y2 - y1))
                    kp_coords.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    kp_coords.append(None)

            # Draw skeleton
            for i, j in SKELETON:
                if kp_coords[i] and kp_coords[j]:
                    cv2.line(frame, kp_coords[i], kp_coords[j], (255, 0, 0), 2)

    # Show output
    cv2.imshow("YOLOv8n + MoveNet Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
