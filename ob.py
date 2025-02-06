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
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=announce, daemon=True).start()

# Webcam setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Higher width
cap.set(4, 480)  # Higher height

# YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", 
    "door", "window", "lamp", "fan", "television", "desk", "wardrobe", "mirror", "blanket", "pillow",
    "shoes", "slippers", "water bottle", "pen", "pencil", "notebook", "bag", "watch", "glasses", "umbrella",
    "bicycle helmet", "coffee mug", "lunch box", "trash can", "plant pot", "printer", "router", "switchboard",
    "helmet", "hat", "mask", "thermometer", "first aid kit", "torch", "ladder", "basket", "mop", "broom",
    "dustpan", "bucket", "washing machine", "dishwasher", "vacuum cleaner", "iron", "stove", "cooker",
    "tissue box", "soap", "shampoo", "toothpaste", "comb", "brush", "scarf", "gloves", "sunglasses",
    "keys", "wallet", "earphones", "headphones", "speaker", "tablet", "charger", "power bank",
    "toy car", "ball", "drone", "kite", "sandals", "newspaper", "magazine"
]


# MediaPipe Pose model for action detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame_skip = 2  # Adjust frame skipping for balance between performance and accuracy
frame_count = 0
last_person_announcement_time = time.time()
last_object_announcement_time = time.time()

# Define a function to recognize activities based on pose
def detect_activity(landmarks):
    # Example conditions for different activities based on keypoints
    
    # Get keypoints
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Define conditions for activities
    if left_ankle.y > left_knee.y and right_ankle.y > right_knee.y:
        return "Sitting"  # Person is sitting

    elif left_hip.y < right_hip.y and left_knee.y < left_hip.y and right_knee.y < right_hip.y:
        return "Standing"  # Person is standing

    elif abs(left_hip.y - right_hip.y) < 0.1 and abs(left_knee.y - right_knee.y) < 0.1:
        return "Lying Down"  # Person is lying down

    elif left_wrist.y < nose.y and right_wrist.y < nose.y:
        return "Drinking Water"  # Person is drinking water

    elif abs(left_wrist.x - right_wrist.x) < 0.1:
        return "Typing"  # Person is typing

    elif left_wrist.y > nose.y and right_wrist.y > nose.y:
        return "Eating"  # Person is eating

    elif left_hip.x - right_hip.x > 0.5:
        return "Walking"  # Person is walking

    elif left_wrist.x > right_wrist.x and right_wrist.x > left_wrist.x:
        return "Using Phone"  # Person is using phone

    else:
        return "Idle"  # Default action if no known activity is detected

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # YOLO detection
    results = model.predict(img, conf=0.5, iou=0.4)  # Refine confidence and IoU thresholds
    detected_objects = Counter()

    person_activities = []

    # Process YOLO detections
    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            if confidence < 0.5:  # Filter out low-confidence detections
                continue

            # Get bounding box coordinates and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Process person detection for action recognition
            if class_name == "person":
                # Crop person region for pose detection
                person_img = img[y1:y2, x1:x2]
                rgb_person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb_person_img)

                if results_pose.pose_landmarks:
                    # Analyze keypoints for actions
                    action = detect_activity(results_pose.pose_landmarks.landmark)
                    person_activities.append(action)

            # Track detected objects
            detected_objects[class_name] += 1

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{class_name} {confidence*100:.1f}%",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Announce detected objects every 5 seconds
    current_time = time.time()

    if current_time - last_object_announcement_time >= 5:
        for obj, count in detected_objects.items():
            if obj != "person":  # Exclude persons from the object list
                announcement_queue.put(f"{obj} detected {count} time(s).")
        last_object_announcement_time = current_time

    # Announce detected activities for persons every 10 seconds
    if current_time - last_person_announcement_time >= 10:
        if person_activities:
            for activity in person_activities:
                announcement_queue.put(f"Person is {activity}.")
        else:
            announcement_queue.put("Person is idle.")
        last_person_announcement_time = current_time
        person_activities.clear()

    # Display output
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
