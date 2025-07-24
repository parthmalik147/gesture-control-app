# src/data_collection/collector.py
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

from src.utils.console_utils import clear_screen, print_colored, progress_bar
from src.utils.camera_utils import extract_hand_landmarks, draw_hand_landmarks, get_available_cameras
from src.config import get_config
from src.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_CYAN, COLOR_BOLD, COLOR_WHITE

def start_gathering():
    """
    Guides the user through collecting hand gesture data for each defined action.
    """
    config = get_config()
    data_dir = config['data_dir']
    actions = config['actions']
    num_samples_per_action = config['num_samples_per_action']
    camera_index = config['camera_index']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check for available cameras before starting data collection
    available_cams = get_available_cameras()
    if not available_cams:
        print_colored("No cameras found. Cannot proceed with data collection.", COLOR_RED)
        time.sleep(2)
        return

    if camera_index not in available_cams:
        print_colored(f"Configured camera index {camera_index} not available. Trying first available camera.", COLOR_YELLOW)
        camera_index = available_cams[0]
        config['camera_index'] = camera_index # Update config if changed
        # config needs to be saved if it's modified in memory
        # save_config() # Decide if you want to save camera_index change immediately or later.
                        # For now, we'll let main menu handle saving on exit from settings.

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print_colored(f"Error: Could not open camera at index {camera_index}. Please check camera connection or index.", COLOR_RED)
        time.sleep(2)
        return

    with mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        for action_idx, action in enumerate(actions):
            action_data_dir = os.path.join(data_dir, action)
            os.makedirs(action_data_dir, exist_ok=True)

            print_colored(f"\n--- Gathering data for: {action} ---", COLOR_CYAN + ";" + COLOR_BOLD)
            print_colored(f"Prepare to perform '{action}' gesture {num_samples_per_action} times.", COLOR_WHITE)
            print_colored("Press 's' to start when ready. Press 'q' to quit at any time.", COLOR_YELLOW)

            start_collection = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    print_colored("Failed to grab frame. Exiting.", COLOR_RED)
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                frame = draw_hand_landmarks(frame, results)

                # Display instructions and progress
                if not start_collection:
                    cv2.putText(frame, f"ACTION: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Press 's' to START / 'q' to QUIT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"ACTION: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Samples: {len(os.listdir(action_data_dir))}/{num_samples_per_action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


                cv2.imshow('Gesture Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print_colored("Data collection aborted by user.", COLOR_RED)
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s') and not start_collection:
                    start_collection = True
                    print_colored("Starting collection...", COLOR_GREEN)
                    time.sleep(1) # Give a moment to prepare
                    clear_screen() # Clear console for progress bar

                if start_collection:
                    # Collect data only if hand is detected
                    landmarks = extract_hand_landmarks(results)
                    if landmarks is not None:
                        current_samples = len(os.listdir(action_data_dir))
                        if current_samples < num_samples_per_action:
                            filename = os.path.join(action_data_dir, f'sample_{current_samples:04d}.npy')
                            np.save(filename, landmarks)
                            progress_bar(current_samples + 1, num_samples_per_action, title=f"Collecting for '{action}'")
                        else:
                            print_colored(f"\nFinished collecting for {action}.", COLOR_GREEN)
                            time.sleep(1)
                            break # Move to next action

            clear_screen() # Clear console after each action

    cap.release()
    cv2.destroyAllWindows()
    print_colored("\nAll gesture data collected!", COLOR_GREEN + ";" + COLOR_BOLD)
    time.sleep(2)