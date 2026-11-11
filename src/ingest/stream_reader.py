import cv2

RTMP_URL = "rtmp://localhost:1935/live/stream"

cap=cv2.VideoCapture(RTMP_URL,cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error opening video stream or file")
    print("Please ensure that the RTMP server is running and accessible at the specified URL.")
    exit()

print('stream opened.press 'q' to exit')
print(f'Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}*{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
print(f'FPS:{cap.get(cv2.CAP_PROP_FPS)}')

while True:
    ret,frame=cap.read()
    if not ret:
        print('frame garb failed .Reconnecting')
        cap=cv2.VideoCapture(RTMP_URL,cv2.CAP_FFMPEG)
        continue

    cv2.imshow('stream Reader',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
