# src/gesture_recognition/recognizer.py
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import tensorflow as tf
import screen_brightness_control as sbc # Used for initial brightness check

from src.utils.console_utils import clear_screen, print_colored
from src.utils.camera_utils import extract_hand_landmarks, draw_hand_landmarks, get_available_cameras
from src.gesture_recognition.action_handler import execute_gesture_action
from src.config import get_config
from src.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_CYAN, COLOR_BOLD, COLOR_WHITE, ACTIVE_ACTIONS_FILE_PATH

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands

def activate_gesture_control():
    """
    Activates real-time gesture recognition using the trained model.
    """
    config = get_config()
    model_path = config['model_path']
    confidence_threshold = config['confidence_threshold']
    display_duration_seconds = config['display_duration_seconds']
    camera_index = config['camera_index']

    clear_screen()
    print_colored("--- Activating Real-time Gesture Control ---", COLOR_CYAN + ";" + COLOR_BOLD)

    # Check if model exists
    if not os.path.exists(model_path):
        print_colored("No trained model found. Please train a model first.", COLOR_RED)
        time.sleep(2)
        return

    # Check if active actions file exists
    if not os.path.exists(ACTIVE_ACTIONS_FILE_PATH):
        print_colored("Active actions file not found. Please train a model first.", COLOR_RED)
        time.sleep(2)
        return

    # Load model and active actions
    try:
        model = tf.keras.models.load_model(model_path)
        with open(ACTIVE_ACTIONS_FILE_PATH, 'r') as f:
            actions = json.load(f)
        print_colored(f"Model '{os.path.basename(model_path)}' loaded successfully.", COLOR_GREEN)
        print_colored(f"Recognizing actions: {', '.join(actions)}", COLOR_GREEN)
    except Exception as e:
        print_colored(f"Error loading model or active actions: {e}", COLOR_RED)
        print_colored("Ensure your model and active_actions.json are intact and compatible.", COLOR_YELLOW)
        time.sleep(3)
        return

    # Check for available cameras
    available_cams = get_available_cameras()
    if not available_cams:
        print_colored("No cameras found. Cannot proceed with real-time control.", COLOR_RED)
        time.sleep(2)
        return

    if camera_index not in available_cams:
        print_colored(f"Configured camera index {camera_index} not available. Trying first available camera.", COLOR_YELLOW)
        camera_index = available_cams[0]
        # It's better not to change config here directly unless confirmed by user via settings menu.

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print_colored(f"Error: Could not open camera at index {camera_index}. Please check camera connection or index.", COLOR_RED)
        time.sleep(2)
        return

    # Initial brightness value for reference
    current_brightness_val = 50 # Default if SBC fails or not present
    try:
        current_brightness_list = sbc.get_brightness()
        if current_brightness_list:
            current_brightness_val = current_brightness_list[0]
    except Exception:
        pass # Ignore if SBC not available or fails


    last_action_display_time = 0
    display_message = ""
    fps_start_time = time.time()
    frame_count = 0

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        print_colored("\nReal-time gesture control active. Press 'q' to quit.", COLOR_WHITE + ";" + COLOR_BOLD)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print_colored("Failed to grab frame. Exiting real-time control.", COLOR_RED)
                break

            frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if config['show_landmarks_in_live_mode']:
                frame = draw_hand_landmarks(frame, results)

            prediction_label = "No Hand Detected"
            prediction_confidence = 0.0

            landmarks = extract_hand_landmarks(results)
            if landmarks is not None:
                # Reshape for model prediction (batch_size, num_landmarks * 3)
                prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)[0]
                predicted_class_index = np.argmax(prediction)
                prediction_confidence = prediction[predicted_class_index]
                prediction_label = actions[predicted_class_index]

                if prediction_confidence > confidence_threshold:
                    # Execute action and get potential display message
                    action_result, action_time, updated_brightness = execute_gesture_action(prediction_label, config, current_brightness_val)
                    if action_result:
                        display_message = action_result
                        last_action_display_time = action_time
                        current_brightness_val = updated_brightness # Update brightness state if adjusted

            # Display messages
            if time.time() - last_action_display_time < display_duration_seconds and display_message:
                cv2.putText(frame, display_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Gesture: {prediction_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


            if config['show_confidence']:
                cv2.putText(frame, f"Conf: {prediction_confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            # Calculate and display FPS
            frame_count += 1
            if time.time() - fps_start_time >= 1: # Update FPS every second
                fps = frame_count / (time.time() - fps_start_time)
                if config['show_fps']:
                    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                fps_start_time = time.time()
                frame_count = 0

            cv2.imshow('Gesture Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print_colored("Exiting real-time gesture control.", COLOR_RED)
                break

    cap.release()
    cv2.destroyAllWindows()
    print_colored("Real-time gesture control session ended.", COLOR_GREEN)
    time.sleep(2)