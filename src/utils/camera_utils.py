# src/utils/camera_utils.py
import cv2
import mediapipe as mp
import numpy as np
import sys
import os

from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_available_cameras():
    """
    Checks and returns a list of available camera indices.
    """
    available_cameras = []
    # Try indices from 0 to 9 to find available cameras
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    if not available_cameras:
        print_colored("Warning: No cameras found. Please ensure a webcam is connected.", COLOR_YELLOW)
    return available_cameras

def extract_hand_landmarks(results):
    """
    Extracts 3D hand landmark coordinates from MediaPipe results.
    Returns a flattened numpy array of 21*3 (x,y,z) coordinates, or None if no hand is detected.
    """
    if results.right_hand_landmarks:
        # Flatten the 21 landmarks (x, y, z) into a single 63-element array
        # Normalize landmarks to the first landmark (wrist) to make them translation-invariant
        landmarks = []
        wrist_x = results.right_hand_landmarks.landmark[0].x
        wrist_y = results.right_hand_landmarks.landmark[0].y
        wrist_z = results.right_hand_landmarks.landmark[0].z

        for landmark in results.right_hand_landmarks.landmark:
            landmarks.append(landmark.x - wrist_x)
            landmarks.append(landmark.y - wrist_y)
            landmarks.append(landmark.z - wrist_z)
        return np.array(landmarks)
    return None

def draw_hand_landmarks(image, results):
    """
    Draws hand landmarks and connections on the image.
    """
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green landmarks
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Red connections
        )
    return image