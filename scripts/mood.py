import os
import cv2
import numpy as np
import mediapipe as mp

import pickle
from PIL import Image
from io import BytesIO

import streamlit as st

# Load the mediapipe for the face detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Select model typr
model_type = st.selectbox(
    'Select Model Type',
    ['Binary Mood Detection', 'Multi-Class Mood Detection'],
)

model_path = (
    'models/binary_mood_detection.pkl'
    if model_type == 'Binary Mood Detection'
    else 'models/mood_detection.pkl'
)

# Load the model
model_path = os.path.join('models', 'binary_mood_detection.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Read the image from camera input
image_buffer = st.camera_input('Capture a photo')

# Convert the image to RGB
if image_buffer is not None:

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        image = np.array(Image.open(image_buffer))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict the mood
        results = holistic.process(image_rgb)

        # Crop face
        if results.face_landmarks is not None:
            face_landmarks = results.face_landmarks.landmark

            height, width, _ = image.shape

            all_landmarks = np.array([
                [
                    landmark.x,
                    landmark.y,
                ] for landmark in results.face_landmarks.landmark
            ])

            # Get the bounding box
            x_min, x_max = all_landmarks[:, 0].min(), all_landmarks[:, 0].max()
            y_min, y_max = all_landmarks[:, 1].min(), all_landmarks[:, 1].max()

            # Convert the bounding box to pixel
            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)

            # Crop the face
            face = image[y_min:y_max, x_min:x_max]

            st.image(face)

        # Then use the image to predict face again
        image_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        results = holistic.process(image_rgb)

        if results.face_landmarks is None:
            st.write('No face detected')

        features = np.array([
            [
                landmark.x,
                landmark.y,
                landmark.z
            ] for landmark in results.face_landmarks.landmark
        ]).flatten()

        mood = model.predict([features])

        st.write('Mood:', mood[0])