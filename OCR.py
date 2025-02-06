import pytesseract
import cv2
import pyttsx3
import os

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Adjust speech speed for clarity

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)  # Set width
webcam.set(4, 480)  # Set height

# Print initial prompt
print("Press 'z' to capture an image and process text.")
print("Press 'q' to exit the program.")

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = webcam.read()

    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Display the video feed
    cv2.imshow("Live Video Feed", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF  # Use bitwise AND for cross-platform compatibility

    if key == ord('z'):  # Press 'z' to capture an image
        print("Image captured! Processing for text...")

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Apply noise reduction
        processed_image = cv2.medianBlur(thresh, 3)

        # Apply dilation and erosion to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed_image = cv2.dilate(processed_image, kernel, iterations=1)
        processed_image = cv2.erode(processed_image, kernel, iterations=1)

        # Use pytesseract to extract text with configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config)

        # Print the extracted text
        print("Extracted Text: ", text)

        # Use text-to-speech to read the text aloud
        engine.say(text)
        engine.runAndWait()

    elif key == ord('q'):  # Press 'q' to exit the program
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
