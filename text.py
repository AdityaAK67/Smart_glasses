# import cv2
# import easyocr
# import pyttsx3

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()
# tts_engine.setProperty('rate', 150)  # Adjust speech speed

# # Initialize EasyOCR Readers
# reader_group1 = easyocr.Reader(['hi', 'mr'], gpu=True)  # Hindi, Marathi
# reader_group2 = easyocr.Reader(['bn', 'as', 'en'], gpu=True)  # Bengali, Assamese, English
# # reader_group3 = easyocr.Reader(['ta', 'te', 'kn', 'en'], gpu=False)  # Tamil, Telugu, Kannada, English

# # Function to process the captured frame and convert text to speech
# def process_frame(frame):
#     text1 = reader_group1.readtext(frame)  # Read text from Group 1
#     text2 = reader_group2.readtext(frame)  # Read text from Group 2
#     # text3 = reader_group3.readtext(frame)  # Read text from Group 3

#     detected_texts = []
#     threshold = 0.25

#     # Process texts from all groups
#     for text_data in (text1 + text2 ):
#         bbox, text, score = text_data
#         if score > threshold:
#             detected_texts.append(text)
#             cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
#             cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

#     if detected_texts:
#         text_to_speak = " ".join(detected_texts)
#         print("Detected Text:", text_to_speak)
#         tts_engine.say(text_to_speak)
#         tts_engine.runAndWait()

#     return frame

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imshow('Live Feed', frame)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('z'):  # Capture frame and detect text when 'z' is pressed
#         processed_frame = process_frame(frame.copy())
#         cv2.imshow('Processed Frame', processed_frame)

#     elif key == ord('q'):  # Quit when 'q' is pressed
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import easyocr
import pyttsx3
import threading

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed

# Initialize EasyOCR Readers (Use GPU for faster processing)
reader_group1 = easyocr.Reader(['hi', 'mr'], gpu=True)  # Hindi, Marathi
reader_group2 = easyocr.Reader(['bn', 'as', 'en'], gpu=True)  # Bengali, Assamese, English

# Function to process the captured frame and convert text to speech
def process_frame(frame):
    # Resize the frame to speed up processing
    frame_resized = cv2.resize(frame, (640, 480))  # Resize for faster OCR

    # Perform OCR only on the resized frame
    text1 = reader_group1.readtext(frame_resized)  # Read text from Group 1
    text2 = reader_group2.readtext(frame_resized)  # Read text from Group 2

    detected_texts = []
    threshold = 0.25  # Confidence threshold

    # Process texts from both groups
    for text_data in (text1 + text2):
        bbox, text, score = text_data
        if score > threshold:
            detected_texts.append(text)
            # Draw bounding box and text on the frame
            cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # Only speak if detected text exists
    if detected_texts:
        text_to_speak = " ".join(detected_texts)
        print("Detected Text:", text_to_speak)
        # Use threading to run TTS in parallel
        threading.Thread(target=speak_text, args=(text_to_speak,)).start()

    return frame

# Function to speak text asynchronously
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live feed
    cv2.imshow('Live Feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('z'):  # Capture frame and detect text when 'z' is pressed
        processed_frame = process_frame(frame.copy())
        cv2.imshow('Processed Frame', processed_frame)

    elif key == ord('q'):  # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
