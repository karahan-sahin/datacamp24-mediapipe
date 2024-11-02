import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import streamlit as st

# Add title and description
st.title("Rock Paper Scissors Game")
st.write("Take a picture of your hand gesture to play Rock Paper Scissors")

base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Create gesture mapping
gesture_mapping = {
    'Closed_Fist': '✊',
    'Open_Palm': '✋',
    'Victory': '✌️'
}

# Load the mediapipe holistic model
img_file_buffer = st.camera_input("",)

if img_file_buffer is not None:

    # Generate the random gesture
    auto_gesture = np.random.choice([
        '✊',
        '✋',
        '✌️'
    ])

    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    img_array = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    # To write image file buffer to disk:
    gesture = recognizer.recognize(img_array)

    if gesture.gestures is None:
        st.error("No gesture detected. Please retake the picture")
        st.stop()
    
    gesture_name = gesture.gestures[0][0].category_name

    if not gesture_name in gesture_mapping:
        # Retake the picture
        st.error("Gesture not recognized. Please retake the picture")
        st.stop()

    gesture_name = gesture_mapping[gesture_name]
        
    st.info(f"You selected {gesture_name}")
    st.info(f"Model selected {auto_gesture}")

    if gesture_name == auto_gesture:
        st.success("Tie")
    elif gesture_name == '✊' and auto_gesture == '✌️':
        st.success("You Win")
    elif gesture_name == '✋' and auto_gesture == '✊':
        st.success("You Win")
    elif gesture_name == '✌️' and auto_gesture == '✋':
        st.success("You Win")
    else:
        st.error("You Lose")
