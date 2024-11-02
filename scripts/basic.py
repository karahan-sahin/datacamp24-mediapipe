import os
import cv2
import numpy as np
import mediapipe as mp

# Load the mediapipe pose model
mp_pose = mp.solutions.pose

# Load the image
image_path = os.path.join('images', 'guitar.jpeg')

# Read the image
image = cv2.imread(image_path)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the pose model
pose = mp_pose.Pose()

# Process the image
results = pose.process(image_rgb)

# Print the results
print('Pose Landmarks:', results.pose_landmarks)

# Write pose landmarks on the image
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Save the image
output_path = os.path.join('output', 'guitar_pose.jpg')
cv2.imwrite(output_path, image)

# Or use drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Draw the pose landmarks on the image
image_with_landmarks = image.copy()
# Or create a blank image
blank_image = np.zeros(image.shape, dtype=np.uint8)
# Make the blank image white
blank_image.fill(255)

mp_drawing.draw_landmarks(
    blank_image,
    results.pose_landmarks, 
    mp_pose.POSE_CONNECTIONS
)

# Save the image
output_path = os.path.join('output', 'guitar_pose_landmarks.jpg')
cv2.imwrite(output_path, blank_image)