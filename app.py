import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from ultralytics import YOLO
import joblib

app = Flask(__name__)

# ======================
# Load YOLOv8 model
# ======================
yolo = YOLO("yolov8n.pt")

# ======================
# Load MoveNet (Pose Estimation)
# ======================
interpreter = tf.lite.Interpreter(model_path="4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_idx = input_details[0]["index"]
output_idx = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
input_size = input_details[0]["shape"][1]  # usually 192 or 256

# ======================
# Load classifier + encoder
# ======================
clf = joblib.load("xgb_posture_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# ======================
# Skeleton connections
# ======================
SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),     # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),         # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ======================
# Helper functions
# ======================
def run_movenet(img):
    """Run MoveNet on a cropped person image"""
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


def extract_features_from_keypoints(keypoints_dict):
    """Compute the 10 engineered features expected by the classifier"""
    required = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for r in required:
        if r not in keypoints_dict:
            return None

    def dist(a, b):
        ax, ay = keypoints_dict[a][:2]
        bx, by = keypoints_dict[b][:2]
        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def angle3(a, b, c):
        ax, ay = keypoints_dict[a][:2]
        bx, by = keypoints_dict[b][:2]
        cx, cy = keypoints_dict[c][:2]
        ab = np.array([ax - bx, ay - by])
        cb = np.array([cx - bx, cy - by])
        dot = np.dot(ab, cb)
        mag = np.linalg.norm(ab) * np.linalg.norm(cb)
        return np.arccos(dot / mag) if mag > 1e-6 else 0.0

    lsx, lsy = keypoints_dict["left_shoulder"][:2]
    rsx, rsy = keypoints_dict["right_shoulder"][:2]
    shoulder_mid = np.array([(lsx + rsx) / 2, (lsy + rsy) / 2])

    lhx, lhy = keypoints_dict["left_hip"][:2]
    rhx, rhy = keypoints_dict["right_hip"][:2]
    hip_mid = np.array([(lhx + rhx) / 2, (lhy + rhy) / 2])

    torso_length = dist("left_shoulder", "left_hip")
    if torso_length == 0:
        torso_length = 1e-6
    normalize = lambda v: v / torso_length

    nx, ny = keypoints_dict["nose"][:2]
    head_forward_x = normalize(nx - shoulder_mid[0])
    head_forward_y = normalize(ny - shoulder_mid[1])
    neck_angle = angle3("left_shoulder", "nose", "right_shoulder")

    spine_tilt = np.arctan2(shoulder_mid[1] - hip_mid[1], shoulder_mid[0] - hip_mid[0])
    shoulder_forward_offset = normalize(shoulder_mid[0] - hip_mid[0])
    hip_tilt = np.arctan2(rhy - lhy, rhx - lhx)
    trunk_to_hip_angle = angle3("nose", "left_hip", "right_hip")
    torso_lean_x = normalize(shoulder_mid[0] - hip_mid[0])
    torso_lean_y = normalize(shoulder_mid[1] - hip_mid[1])
    com_x = (shoulder_mid[0] + hip_mid[0]) / 2
    com_shift_x = normalize(com_x - hip_mid[0])

    return np.array([[
        head_forward_x,
        head_forward_y,
        neck_angle,
        spine_tilt,
        shoulder_forward_offset,
        hip_tilt,
        trunk_to_hip_angle,
        torso_lean_x,
        torso_lean_y,
        com_shift_x
    ]], dtype=np.float32)


def generate_frames():
    """Generator that streams YOLO + MoveNet annotated frames with posture classification"""
    cap = cv2.VideoCapture(0)  # webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons with YOLO
        results = yolo(frame, conf=0.5, classes=[0], verbose=False)

        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            keypoints = run_movenet(person_crop)

            # Map to COCO names
            keypoints_dict = {}
            for i, name in enumerate(COCO_KEYPOINTS):
                yv, xv, conf = keypoints[i]
                if conf > 0.3:
                    keypoints_dict[name] = (xv, yv, conf)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw keypoints + skeleton
            kp_coords = []
            for i, (yv, xv, conf) in enumerate(keypoints):
                if conf > 0.3:
                    cx = int(x1 + xv * (x2 - x1))
                    cy = int(y1 + yv * (y2 - y1))
                    kp_coords.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    kp_coords.append(None)

            for (i, j) in SKELETON:
                if kp_coords[i] and kp_coords[j]:
                    cv2.line(frame, kp_coords[i], kp_coords[j], (255, 0, 0), 2)

            # === Predict posture ===
            features = extract_features_from_keypoints(keypoints_dict)
            if features is not None:
                y_pred_encoded = clf.predict(features)[0]
                label = encoder.inverse_transform([y_pred_encoded])[0]
                cv2.putText(frame, f"Posture: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
