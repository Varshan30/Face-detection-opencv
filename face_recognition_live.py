import cv2
import os
import numpy as np

# Load trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to dataset (used to label names)
dataset_path = 'D:\\Projects\\Face project\\datasets'
people = os.listdir(dataset_path)

# Start webcam
cap = cv2.VideoCapture(0)

print("🔍 Starting webcam. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Predict using recognizer
        label, confidence = recognizer.predict(roi_gray)

        # Lower confidence is better (threshold can be adjusted)
        if confidence < 70:
            name = people[label]
            color = (0, 255, 0)  # Green
            label_text = f"{name} ({int(confidence)})"
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red
            label_text = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
