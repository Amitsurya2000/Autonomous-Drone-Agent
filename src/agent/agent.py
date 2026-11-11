from ultralytics import YOLO
import cv2
import time
import datetime
import os
from collections import defaultdict

# --- CONFIGURATION ---
# 1. Use the RTMP stream from MediaMTX
VIDEO_SOURCE = "rtmp://localhost:1935/live/stream"

MIN_CONF = 0.5
TRIGGER_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"} 
COOLDOWN_SEC = 10.0 

# 2. Save directly to data/crops so edge_uploader.py sees it
CROPS_DIR = "data/crops"
os.makedirs(CROPS_DIR, exist_ok=True)

# --- 3. THE MAIN LOOP (THE WAITER) ---
print("[MAIN] Loading YOLO (The Eyes)...")
yolo_model = YOLO("yolo11n.pt")

# 4. Crucial: Include cv2.CAP_FFMPEG so it can read the RTMP stream
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("[ERROR] Cannot open RTMP stream. Check MediaMTX and DJI GO 4 app.")
    exit()

last_trigger = defaultdict(float)
print("[MAIN] System running! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame grab failed. Reconnecting...")
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        continue

    results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    annotated = results.plot()
    now = time.time()
    
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes
        for i in range(len(boxes)):
            cls_name = yolo_model.names[int(boxes.cls[i])]
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i])

            # Gate Rules
            if conf < MIN_CONF or cls_name not in TRIGGER_CLASSES:
                continue
            if now - last_trigger[track_id] < COOLDOWN_SEC:
                continue

            # TRIGGER EVENT!
            last_trigger[track_id] = now
            
            # Get precise timestamp for the filename
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Draw thick red box
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Crop the object from the frame
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            
            if crop.size > 0:
                # Save to data/crops (edge_uploader.py will grab this instantly)
                filename = f"{CROPS_DIR}/drone_{current_time}_{cls_name}_id{track_id}.jpg"
                cv2.imwrite(filename, crop)
                print(f"[{current_time}] [MAIN] Event Saved: {filename}")

    # Show live video
    cv2.putText(annotated, "Drone Edge Agent Running", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Agent Pipeline", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()