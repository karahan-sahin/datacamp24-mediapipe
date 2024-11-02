import os
import cv2
import numpy as np
import mediapipe as mp

# Load the mediapipe hands model
mp_hands = mp.solutions.hands

# Load the image
image_path = os.path.join('images', 'hands.jpg')

# Read the image
image = cv2.imread(image_path)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the hands model
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=5,
    min_detection_confidence=0.1
)

# Process the image
results = hands.process(image)

# Print the results
print('Hand Landmarks:', results.multi_hand_landmarks)

mp_drawing = mp.solutions.drawing_utils
# Write hand landmarks on the image
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

# Save the image
output_path = os.path.join('output', 'hand_landmarks.jpg')
cv2.imwrite(output_path, image)
