import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import time
import threading

import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from joint_mapping_2 import XHandPublisher

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Shared result + lock for thread safety
latest_result = None
result_lock = threading.Lock()
def draw_landmarks_on_frame(frame, result):
    h, w, _ = frame.shape
    if not result.hand_landmarks:
        return frame

    # MediaPipe hand connections
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),         # Thumb
        (0,5),(5,6),(6,7),(7,8),         # Index
        (0,9),(9,10),(10,11),(11,12),    # Middle
        (0,13),(13,14),(14,15),(15,16),  # Ring
        (0,17),(17,18),(18,19),(19,20),  # Pinky
        (5,9),(9,13),(13,17),            # Palm
    ]

    for hand_landmarks in result.hand_landmarks:
        # Convert normalised coords to pixel coords
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        # Draw landmark dots
        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return frame

xhand = XHandPublisher(host='localhost', port=9090, kp=100, ki=0, kd=1200, effort_limit=100, mode=3)

def store_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result
    if result.hand_landmarks:
        xhand.send(result.hand_landmarks[0])

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.9,  # default is 0.5
    result_callback=store_result)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Draw latest available result onto the frame
            with result_lock:
                result_to_draw = latest_result
            if result_to_draw:
                frame = draw_landmarks_on_frame(frame, result_to_draw)

            cv2.imshow("Hand Tracking", frame)

            try:
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cap.release()
        cv2.destroyAllWindows()