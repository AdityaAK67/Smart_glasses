from ultralytics import YOLO
import cv2
import pyttsx3
import mediapipe as mp
import threading
from queue import Queue
import time
from collections import Counter

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

announcement_queue = Queue()

def announce():
    while True:
        text = announcement_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()

threading.Thread(target=announce, daemon=True).start()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame_skip = 2  # Skip frames for performance
last_person_announcement_time = time.time()
last_object_announcement_time = time.time()

def detect_activity(landmarks):
    """Detect activity based on pose keypoints."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    if left_ankle.y > left_hip.y and right_ankle.y > right_hip.y:
        return "Sitting"
    elif abs(left_hip.y - right_hip.y) < 0.1:
        return "Standing"
    return "Idle"

def object_detection(img):
    """Perform object detection using YOLO."""
    results = model.predict(img, conf=0.5, iou=0.4)
    detected_objects = Counter()
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detected_objects[class_name] += 1
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return results, detected_objects, img

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    
    results, detected_objects, img = object_detection(img)
    
    # Announce objects every 5 seconds
    if time.time() - last_object_announcement_time >= 5:
        for obj, count in detected_objects.items():
            if obj != "person":
                announcement_queue.put(f"{obj} detected {count} time(s).")
        last_object_announcement_time = time.time()
    
    # Pose detection for persons
    person_activities = []
    for obj in detected_objects:
        if obj == "person":
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_img)
            if results_pose.pose_landmarks:
                action = detect_activity(results_pose.pose_landmarks.landmark)
                person_activities.append(action)
    
    # Announce person activities every 10 seconds
    if time.time() - last_person_announcement_time >= 10:
        if person_activities:
            for activity in set(person_activities):
                announcement_queue.put(f"Person is {activity}.")
        else:
            announcement_queue.put("Person is idle.")
        last_person_announcement_time = time.time()
    
    # Display video feed
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
