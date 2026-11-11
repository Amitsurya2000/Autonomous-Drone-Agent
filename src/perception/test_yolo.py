from ultralytics import YOLO
import cv2
import time

# 1. CHANGED: Use webcam (0) instead of RTMP
VIDEO_SOURCE = 0 

# Load model once
model = YOLO("yolo11n.pt")
print(f"[OK] YOLO loaded with {len(model.names)} classes")

# 2. CHANGED: Open the webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam. Is it being used by another app?")
    exit()

print("[OK] Webcam opened. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame grab failed.")
        continue # Webcams don't need the heavy reconnect logic

    # YOLO inference on this frame
    results = model(frame, verbose=False)[0]

    # Draw boxes on frame
    annotated = results.plot()

    # FPS calc
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    cv2.putText(annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detections: {len(results.boxes)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO on Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()