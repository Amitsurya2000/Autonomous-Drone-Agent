from ultralytics import YOLO
import cv2
import time
import os
from collections import defaultdict

# Use 0 for webcam
VIDEO_SOURCE = 0 

# --- GATE CONFIGURATION ---
MIN_CONF = 0.5
TRIGGER_CLASSES = {"person", "cell phone", "cup", "bottle"} # The objects we care about
COOLDOWN_SEC = 5.0 # Don't trigger the same object more than once every 5s

# Path setup
CROP_DIR = "data/crops"
if not os.path.exists(CROP_DIR):
    os.makedirs(CROP_DIR)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

print("[OK] Event Gate started. Press 'q' to quit.")
last_trigger = defaultdict(float)
events_this_session = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    annotated = results.plot()

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    triggered = []

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = model.names[cls_id]
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i])

            # Gate Rules
            if conf < MIN_CONF: continue
            if cls_name not in TRIGGER_CLASSES: continue
            if now - last_trigger[track_id] < COOLDOWN_SEC: continue

            # If it passes rules -> TRIGGER!
            last_trigger[track_id] = now
            events_this_session += 1
            triggered.append(cls_name)

            # Crop the image
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Save the crop with a unique name
                fname = f"{CROP_DIR}/track{track_id}_{int(now)}_{cls_name}.jpg"
                cv2.imwrite(fname, crop)
                print(f"[EVENT] {cls_name} (ID: {track_id}) -> Saved: {fname}")

            # Draw a red box to show it triggered
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.putText(annotated, f"FPS: {fps:.1f}  Events: {events_this_session}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if triggered:
        cv2.putText(annotated, "TRIGGERED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Event Gate", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()