import cv2
import os
import numpy as np

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define dataset path
dataset_path = 'D:\\Projects\\Face project\\datasets'
people = os.listdir(dataset_path)

faces = []
labels = []

# Supported image extensions
valid_extensions = ['.jpg', '.jpeg', '.png']

for label, person in enumerate(people):
    person_path = os.path.join(dataset_path, person)
    for filename in os.listdir(person_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue  # skip non-image files like desktop.ini

        image_path = os.path.join(person_path, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        faces.append(img)
        labels.append(label)

# Final check before training
if len(faces) == 0:
    print("No valid faces found in the dataset.")
else:
    print(f"Found {len(faces)} valid face images. Training now...")
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognizer.yml')
    print("✅ Training complete. Model saved as 'face_recognizer.yml'")
