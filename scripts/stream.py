from pathlib import Path

import os
import cv2
import numpy as np
import mediapipe as mp

import streamlit as st

# Load the mediapipe holistic model
mp_holistic = mp.solutions.holistic


# Initialize the holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the video
video_path = Path('videos', 'sign.mp4')

print(video_path)
# Read the video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    is_success, frame = cap.read()

    if not is_success:
        st.write('Camera not found')
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    results = holistic.process(image_rgb)

    # Print the results
    st.write('Face Landmarks:', results.face_landmarks)
    st.write('Pose Landmarks:', results.pose_landmarks)
    st.write('Left Hand Landmarks:', results.left_hand_landmarks)

    # Draw the face landmarks on the image
    if results.face_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACE_CONNECTIONS
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Display the image
    st.image(frame, channels='BGR')