import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import sys
import os
import subprocess
import warnings
import json # For configuration
from datetime import datetime # For timestamps in logs/feedback
import shutil # For deleting directories

# --- Suppress all warnings ---
#warnings.filterwarnings("ignore") # Keeping this commented for debugging purposes as discussed

# --- Global Constants & Configuration ---
# Determine Application Data Paths
if getattr(sys, 'frozen', False):
    APP_DATA_BASE_DIR = os.path.join(os.path.expanduser('~'), '.gesture_control_app_data')
else:
    APP_DATA_BASE_DIR = os.path.abspath(os.path.dirname(__file__))

os.makedirs(APP_DATA_BASE_DIR, exist_ok=True)

# Configuration File Path
CONFIG_FILE = os.path.join(APP_DATA_BASE_DIR, 'config.json')
ACTIVE_ACTIONS_FILE = os.path.join(APP_DATA_BASE_DIR, 'active_actions.json')


# Default Configuration
DEFAULT_CONFIG = {
    'actions': ['brightness_up', 'brightness_down', 'neutral',
                'volume_up', 'volume_down', 'scroll_up', 'scroll_down',
                'open_browser', 'minimize_window',
                'play_pause', 'next_track', 'prev_track'],
    'num_samples_per_action': 300,
    'confidence_threshold': 0.98,
    'cooldown_seconds': 0.5,
    'display_duration_seconds': 2.0,
    'camera_index': 0, # Default camera
    'show_fps': True,
    'show_confidence': True,
    'show_landmarks_in_live_mode': True
}

CONFIG = {} # Will be loaded from file or use defaults

def load_config():
    """Loads configuration from file or sets defaults."""
    global CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                # Merge defaults with loaded config, prioritizing loaded values
                CONFIG = {**DEFAULT_CONFIG, **loaded_config}
            print_colored(f"Configuration loaded from {CONFIG_FILE}", COLOR_GREEN)
        except json.JSONDecodeError:
            print_colored(f"Error reading config file. Using default settings.", COLOR_RED)
            CONFIG = DEFAULT_CONFIG.copy()
    else:
        CONFIG = DEFAULT_CONFIG.copy()
        save_config() # Save default config if not exists
        print_colored(f"No config file found. Created default config at {CONFIG_FILE}", COLOR_YELLOW)
    # Ensure DATA_DIR and MODEL_PATH are derived from the determined base directory
    CONFIG['data_dir'] = os.path.join(APP_DATA_BASE_DIR, 'simple_gesture_data')
    CONFIG['model_path'] = os.path.join(APP_DATA_BASE_DIR, 'simple_gesture_model.h5')
    os.makedirs(CONFIG['data_dir'], exist_ok=True) # Ensure data directory exists

def save_config():
    """Saves the current configuration to file."""
    try:
        # Create a mutable copy and remove derived paths for clean saving
        config_to_save = {k: v for k, v in CONFIG.items() if k not in ['data_dir', 'model_path']}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print_colored(f"Configuration saved to {CONFIG_FILE}", COLOR_GREEN)
    except Exception as e:
        print_colored(f"Error saving configuration: {e}", COLOR_RED)

# MediaPipe setup (common for all modes)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Console Aesthetics and Utilities ---
def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color_code):
    """Prints text with ANSI color codes."""
    print(f"\033[{color_code}m{text}\033[0m")

# ANSI color codes
COLOR_BLUE = "94"
COLOR_CYAN = "96"
COLOR_GREEN = "92"
COLOR_YELLOW = "93"
COLOR_RED = "91"
COLOR_MAGENTA = "95"
COLOR_WHITE = "97"
COLOR_BOLD = "1"
COLOR_RESET = "0"

def get_available_cameras():
    """Detects and returns a list of available camera indices."""
    available_cameras = []
    for i in range(5):  # Check up to 5 cameras (can be adjusted)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def progress_bar(current, total, bar_length=40):
    """Generates a text-based progress bar."""
    fraction = current / total
    arrow = int(fraction * bar_length) * '#'
    padding = int(bar_length - len(arrow)) * '-'
    return f"[{arrow}{padding}] {int(fraction*100)}%"

# --- System Control Functions ---

# Brightness Control (cross-platform)
try:
    import screen_brightness_control as sbc
    _has_sbc = True
except ImportError:
    _has_sbc = False
    print_colored("Warning: 'screen_brightness_control' not found. Brightness control disabled.", COLOR_YELLOW)
    print_colored("Install with: pip install screen-brightness-control", COLOR_YELLOW)

def adjust_brightness(change_percent, current_brightness_val):
    """Adjusts screen brightness by a given percentage (+/-)."""
    if not _has_sbc:
        return current_brightness_val

    try:
        current_brightness_list = sbc.get_brightness()
        if current_brightness_list:
            current_brightness = current_brightness_list[0]
            new_brightness = max(0, min(100, current_brightness + change_percent))
            sbc.set_brightness(new_brightness)
            return new_brightness
        else:
            return current_brightness_val
    except Exception as e:
        print_colored(f"Brightness adjustment error: {e}", COLOR_RED)
        return current_brightness_val

# PyAutoGUI for cross-platform simulation of keyboard/mouse events
try:
    import pyautogui
    _has_pyautogui = True
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
except ImportError:
    _has_pyautogui = False
    print_colored("Warning: 'pyautogui' not found. Volume, scroll, browser, minimize, media controls disabled.", COLOR_YELLOW)
    print_colored("Install with: pip install pyautogui", COLOR_YELLOW)
except Exception as e:
    _has_pyautogui = False
    print_colored(f"Error initializing pyautogui: {e}", COLOR_RED)


def adjust_volume(direction):
    """Adjusts system volume up or down."""
    if not _has_pyautogui:
        return False

    try:
        if sys.platform == "win32":
            if direction == "up":
                pyautogui.press('volumeup')
            elif direction == "down":
                pyautogui.press('volumedown')
        elif sys.platform == "darwin": # macOS
            # Use osascript for finer control on macOS
            if direction == "up":
                subprocess.run(["osascript", "-e", "set volume output volume (get volume settings)'s output volume + 5"])
            elif direction == "down":
                subprocess.run(["osascript", "-e", "set volume output volume (get volume settings)'s output volume - 5"])
        elif sys.platform.startswith("linux"):
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+" if direction == "up" else "5%-"])
        else:
            print_colored(f"Volume control not supported on {sys.platform}", COLOR_YELLOW)
            return False
        return True
    except Exception as e:
        print_colored(f"Volume control error: {e}", COLOR_RED)
        return False

def scroll_window(direction):
    """Scrolls the active window up or down."""
    if not _has_pyautogui:
        return False

    try:
        if direction == "up":
            pyautogui.scroll(100)
        elif direction == "down":
            pyautogui.scroll(-100)
        return True
    except Exception as e:
        print_colored(f"Scroll error: {e}", COLOR_RED)
        return False

def open_web_browser(url="https://www.google.com"):
    """Opens the default web browser to a specified URL."""
    try:
        import webbrowser
        webbrowser.open(url)
        return True
    except ImportError:
        print_colored("Warning: 'webbrowser' module not found (should be standard).", COLOR_RED)
        return False
    except Exception as e:
        print_colored(f"Browser open error: {e}", COLOR_RED)
        return False

def minimize_active_window():
    """Minimizes the currently active window."""
    if not _has_pyautogui:
        return False

    try:
        if sys.platform == "win32":
            pyautogui.hotkey('win', 'down')
        elif sys.platform == "darwin": # macOS
            pyautogui.hotkey('command', 'm')
        elif sys.platform.startswith("linux"):
            pyautogui.hotkey('alt', 'f9') # Common for Gnome/KDE. May vary.
        else:
            print_colored(f"Minimize window not supported on {sys.platform}", COLOR_YELLOW)
            return False
        return True
    except Exception as e:
        print_colored(f"Minimize window error: {e}", COLOR_RED)
        return False

# --- Media Control Functions ---
def play_pause_media():
    """Toggles play/pause for media."""
    if not _has_pyautogui:
        return False
    try:
        pyautogui.press('playpause')
        return True
    except Exception as e:
        print_colored(f"Play/Pause error: {e}", COLOR_RED)
        return False

def next_track():
    """Skips to the next media track."""
    if not _has_pyautogui:
        return False
    try:
        pyautogui.press('nexttrack')
        return True
    except Exception as e:
        print_colored(f"Next track error: {e}", COLOR_RED)
        return False

def prev_track():
    """Goes to the previous media track."""
    if not _has_pyautogui:
        return False
    try:
        pyautogui.press('prevtrack')
        return True
    except Exception as e:
        print_colored(f"Previous track error: {e}", COLOR_RED)
        return False

# --- Helper Functions for Hand Landmark Processing ---
def extract_hand_landmarks(results):
    """
    Extracts RIGHT hand landmarks (x, y, z) and flattens them.
    Returns a zero array if no RIGHT hand is detected.
    """
    right_hand_landmarks_flat = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label

            if handedness == 'Right':
                right_hand_landmarks_flat = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                break
    return right_hand_landmarks_flat

def draw_hand_landmarks(image, results):
    """Draws all detected hand landmarks (left and right) on the image."""
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness_data in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness_label = handedness_data.classification[0].label

            if handedness_label == 'Right':
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4)
                connection_drawing_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            else: # Left hand
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4)
                connection_drawing_spec = mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec, connection_drawing_spec)

# --- Data Collection Function ---
def start_gathering():
    """Manages the process of collecting gesture data."""
    clear_screen()
    print_colored("--- SELECT ACTIONS FOR DATA COLLECTION ---", COLOR_YELLOW + ";" + COLOR_BOLD)

    # Determine which actions have existing data
    action_data_status = {}
    for action in CONFIG['actions']:
        action_path = os.path.join(CONFIG['data_dir'], action)
        if os.path.exists(action_path) and len(os.listdir(action_path)) > 0:
            action_data_status[action] = "(Trained)"
        else:
            action_data_status[action] = "(No Data)"

    for i, action in enumerate(CONFIG['actions']):
        status = action_data_status.get(action, "")
        print(f"{i+1}. {action} {status}")

    selected_actions_to_collect = []

    while True:
        try:
            sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter numbers (e.g., 1 3), 'all', 'r' for specific retrain, or 'q' to quit: \033[0m")
            sys.stdout.flush()
            choice_input = input().strip().lower()

            if choice_input == 'q':
                return
            elif choice_input == 'all':
                selected_actions_to_collect = CONFIG['actions'][:]
                break
            elif choice_input == 'r':
                print_colored("\n--- RETRAIN/ADD SAMPLES FOR SPECIFIC GESTURE ---", COLOR_YELLOW)
                sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter number of gesture to retrain/add samples for: \033[0m")
                sys.stdout.flush()
                retrain_choice = int(input().strip())
                if 1 <= retrain_choice <= len(CONFIG['actions']):
                    selected_actions_to_collect = [CONFIG['actions'][retrain_choice - 1]]
                    print_colored(f"Selected to retrain/add samples for: {selected_actions_to_collect[0]}", COLOR_CYAN)
                    break
                else:
                    print_colored("Invalid choice. Please enter a valid number.", COLOR_RED)
                    time.sleep(1.5)
                    clear_screen()
                    print_colored("--- SELECT ACTIONS FOR DATA COLLECTION ---", COLOR_YELLOW + ";" + COLOR_BOLD)
                    for i, action in enumerate(CONFIG['actions']):
                        status = action_data_status.get(action, "")
                        print(f"{i+1}. {action} {status}")
                    continue

            choices = [int(c.strip()) for c in choice_input.split()]
            valid_choices = True
            selected_actions_from_input = []
            for choice in choices:
                if 1 <= choice <= len(CONFIG['actions']):
                    selected_actions_from_input.append(CONFIG['actions'][choice - 1])
                else:
                    print_colored(f"Invalid choice: {choice}. Please enter numbers between 1 and {len(CONFIG['actions'])}.", COLOR_RED)
                    valid_choices = False
                    time.sleep(1.5)
                    clear_screen()
                    print_colored("--- SELECT ACTIONS FOR DATA COLLECTION ---", COLOR_YELLOW + ";" + COLOR_BOLD)
                    for i, action in enumerate(CONFIG['actions']):
                        status = action_data_status.get(action, "")
                        print(f"{i+1}. {action} {status}")
                    break
            if valid_choices:
                selected_actions_to_collect = selected_actions_from_input
                break
        except ValueError:
            print_colored("Invalid input. Please enter numbers, 'all', 'r', or 'q'.", COLOR_RED)
            time.sleep(1.5)
            clear_screen()
            print_colored("--- SELECT ACTIONS FOR DATA COLLECTION ---", COLOR_YELLOW + ";" + COLOR_BOLD)
            for i, action in enumerate(CONFIG['actions']):
                status = action_data_status.get(action, "")
                print(f"{i+1}. {action} {status}")

    if not selected_actions_to_collect:
        print_colored("No actions selected. Exiting data collection.", COLOR_RED)
        time.sleep(1.5)
        return

    # Ensure 'neutral' is always included if data collection is happening
    if 'neutral' not in selected_actions_to_collect:
        selected_actions_to_collect.append('neutral')
        print_colored("Note: 'neutral' gesture data will also be collected/updated to maintain balance.", COLOR_YELLOW)
        time.sleep(1)

    # Ensure the DATA_DIR exists before creating action-specific subdirectories
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    for action in selected_actions_to_collect:
        os.makedirs(os.path.join(CONFIG['data_dir'], action), exist_ok=True)

    cap = cv2.VideoCapture(CONFIG['camera_index'])
    if not cap.isOpened():
        print_colored(f"Error: Could not open camera {CONFIG['camera_index']}. Please check if it's in use or try a different camera index in settings.", COLOR_RED)
        time.sleep(3) # Give user time to read error
        return # Return to main menu instead of sys.exit

    print_colored(f"\nStarting data collection. Data will be saved in '{CONFIG['data_dir']}'", COLOR_CYAN)
    print_colored(f"You selected to collect data for: {', '.join(selected_actions_to_collect)}", COLOR_CYAN)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        for action in selected_actions_to_collect:
            clear_screen()
            print_colored(f"\n--- COLLECTING DATA FOR: '{action.upper()}' ---", COLOR_GREEN + ";" + COLOR_BOLD)
            print_colored(f"Perform the '{action}' gesture repeatedly. Press 'q' to stop early.", COLOR_CYAN)
            count = 0

            # Countdown for starting collection
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    print_colored("Failed to grab frame during countdown, exiting...", COLOR_RED)
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                # Visually appealing countdown
                cv2.putText(frame, f'GET READY FOR:', (int(w*0.05), int(h*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'"{action.upper()}"', (int(w*0.05), int(h*0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f'Starting in {i}...', (int(w*0.05), int(h*0.3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Gesture Data Collection', frame)
                cv2.waitKey(700) # Slightly longer pause for readability

            while count < CONFIG['num_samples_per_action']:
                ret, frame = cap.read()
                if not ret:
                    print_colored("Failed to grab frame, exiting...", COLOR_RED)
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                draw_hand_landmarks(image, results)

                landmarks = extract_hand_landmarks(results)
                if not np.all(landmarks == 0):
                    file_path = os.path.join(CONFIG['data_dir'], action, f'{action}_{count}.npy')
                    np.save(file_path, landmarks)
                    count += 1
                    cv2.putText(image, f'COLLECTING: {action.upper()}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Samples: ({count}/{CONFIG["num_samples_per_action"]})',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, progress_bar(count, CONFIG['num_samples_per_action']),
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

                else:
                    cv2.putText(image, "No Right Hand Detected. Adjust hand or lighting.",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Current Action: {action.upper()}',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Gesture Data Collection', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print_colored(f"Stopped collecting for '{action}' early.", COLOR_YELLOW)
                    break

            if not ret: # If camera failed during collection loop
                break

            print_colored(f"Finished collecting {count} samples for '{action}'. Waiting 3 seconds before next action...", COLOR_MAGENTA)
            time.sleep(3)

    cap.release()
    cv2.destroyAllWindows()
    print_colored("\nData collection complete.", COLOR_GREEN + ";" + COLOR_BOLD)
    print_colored(f"Collected data saved in '{CONFIG['data_dir']}' directory.", COLOR_GREEN)
    time.sleep(2)

# --- Model Training Function ---
def train_model():
    """Trains a gesture recognition model using collected data."""
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    clear_screen()
    print_colored("--- TRAINING GESTURE RECOGNITION MODEL ---", COLOR_YELLOW + ";" + COLOR_BOLD)
    print_colored("Loading data...", COLOR_CYAN)

    X = [] # Features (landmark data)
    y = [] # Labels (numerical representation of action)

    # Ensure CONFIG['actions'] is used for mapping
    action_to_idx = {action_name: i for i, action_name in enumerate(CONFIG['actions'])}

    found_data_for_actions = set()

    for action_idx, action_name in enumerate(CONFIG['actions']):
        action_path = os.path.join(CONFIG['data_dir'], action_name)
        if not os.path.exists(action_path):
            continue

        action_samples_count = 0
        for i in range(CONFIG['num_samples_per_action']): # Iterate up to expected num samples
            file_path = os.path.join(action_path, f'{action_name}_{i}.npy')
            if os.path.exists(file_path):
                landmarks = np.load(file_path)
                X.append(landmarks)
                y.append(action_idx)
                action_samples_count += 1
        if action_samples_count > 0:
            found_data_for_actions.add(action_name)
            print(f"  Loaded {action_samples_count} samples for '{action_name}'")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print_colored("No data loaded for training. Please ensure data was collected successfully.", COLOR_RED)
        time.sleep(2)
        return

    # Filter ACTIONS to only include those for which data was found
    active_actions = [action for action in CONFIG['actions'] if action in found_data_for_actions]
    if len(active_actions) < 2: # Need at least two classes for classification
        print_colored("Not enough distinct gesture data (need at least 2 gestures with samples) to train a model.", COLOR_RED)
        print_colored("Please collect data for more gestures.", COLOR_RED)
        time.sleep(3)
        return

    # Remap y if some actions had no data, to ensure correct to_categorical
    # Create a new mapping for only the active actions
    new_action_to_idx = {action_name: i for i, action_name in enumerate(active_actions)}
    re_mapped_y = np.array([new_action_to_idx[CONFIG['actions'][old_idx]] for old_idx in y])

    y_categorical = to_categorical(re_mapped_y, num_classes=len(active_actions)).astype(int)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
    except ValueError as e:
        print_colored(f"\nERROR: Could not split data with stratify=y. {e}", COLOR_RED)
        print_colored("This often means one or more classes have too few samples (e.g., only 1 sample).", COLOR_RED)
        print_colored("Please ensure each selected action has at least 2 collected samples.", COLOR_RED)
        time.sleep(3)
        return

    # Dynamic input shape based on extracted landmarks
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(active_actions), activation='softmax')) # Output layer matches active actions

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping to prevent overfitting and speed up training if not improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print_colored("\nStarting model training (This may take a moment)...", COLOR_CYAN)
    history = model.fit(X_train, y_train, epochs=100, # Increased epochs with early stopping
              validation_data=(X_test, y_test),
              callbacks=[early_stopping],
              verbose=0)

    print_colored("\nModel training complete.", COLOR_GREEN + ";" + COLOR_BOLD)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print_colored(f"Test Loss: {loss:.4f}", COLOR_WHITE)
    print_colored(f"Test Accuracy: {accuracy:.4f}", COLOR_WHITE)

    # Save the model and also the mapping for prediction later
    model.save(CONFIG['model_path'])
    # Save the active_actions list along with the model, or in a separate file
    with open(ACTIVE_ACTIONS_FILE, 'w') as f:
        json.dump(active_actions, f)

    print_colored(f"Model saved as '{CONFIG['model_path']}'", COLOR_GREEN)
    print_colored(f"Active actions mapping saved.", COLOR_GREEN)
    time.sleep(2)

# --- Real-time Gesture Activation Function ---
def activate_gesture_control():
    """Activates real-time gesture control based on the trained model."""
    clear_screen()
    print_colored("--- ACTIVATING REAL-TIME GESTURE CONTROL ---", COLOR_YELLOW + ";" + COLOR_BOLD)
    try:
        model = tf.keras.models.load_model(CONFIG['model_path'])
        # Load the active actions mapping used during training
        with open(ACTIVE_ACTIONS_FILE, 'r') as f:
            active_actions_for_prediction = json.load(f)
        print_colored(f"Successfully loaded model: {CONFIG['model_path']}", COLOR_GREEN)
        print_colored(f"Model trained on: {', '.join(active_actions_for_prediction)}", COLOR_CYAN)
    except Exception as e:
        print_colored(f"Error loading model {CONFIG['model_path']}: {e}", COLOR_RED)
        print_colored("Please ensure you have trained the model by selecting 'n' from the main menu.", COLOR_RED)
        time.sleep(3)
        return

    last_action_time = {action: 0 for action in active_actions_for_prediction}
    current_brightness = 50 if _has_sbc else "N/A" # Initialize brightness value
    _display_action_status = ("", 0.0) # (message, timestamp)
    prev_frame_time = 0

    cap = cv2.VideoCapture(CONFIG['camera_index'])
    if not cap.isOpened():
        print_colored(f"Error: Could not open camera {CONFIG['camera_index']}. Please check if it's in use or try a different camera index in settings.", COLOR_RED)
        time.sleep(3)
        return

    print_colored("\nStarting real-time gesture recognition. Press 'q' to quit.", COLOR_CYAN)
    time.sleep(1)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print_colored("Failed to grab frame, exiting...", COLOR_RED)
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if CONFIG['show_landmarks_in_live_mode']:
                draw_hand_landmarks(image, results)

            predicted_action_label = "No Gesture"
            prediction_confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks = extract_hand_landmarks(results)
                if not np.all(landmarks == 0): # Check if right hand was detected and landmarks extracted
                    landmarks_reshaped = np.expand_dims(landmarks, axis=0)
                    predictions = model.predict(landmarks_reshaped, verbose=0)[0]
                    predicted_action_idx = np.argmax(predictions)
                    prediction_confidence = predictions[predicted_action_idx]

                    if predicted_action_idx < len(active_actions_for_prediction):
                        predicted_action_label = active_actions_for_prediction[predicted_action_idx]
                    else:
                        predicted_action_label = "Unknown Action" # Should not happen if mapping is correct

                    current_time = time.time()
                    if predicted_action_label != 'neutral' and prediction_confidence >= CONFIG['confidence_threshold']:
                        if (current_time - last_action_time.get(predicted_action_label, 0)) > CONFIG['cooldown_seconds']:
                            action_successful = False

                            if predicted_action_label == 'brightness_up':
                                current_brightness = adjust_brightness(10, current_brightness)
                                _display_action_status = (f"Brightness: {current_brightness}%", current_time)
                                action_successful = True
                            elif predicted_action_label == 'brightness_down':
                                current_brightness = adjust_brightness(-10, current_brightness)
                                _display_action_status = (f"Brightness: {current_brightness}%", current_time)
                                action_successful = True
                            elif predicted_action_label == 'volume_up':
                                if adjust_volume("up"):
                                    _display_action_status = ("Volume Up!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'volume_down':
                                if adjust_volume("down"):
                                    _display_action_status = ("Volume Down!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'scroll_up':
                                if scroll_window("up"):
                                    _display_action_status = ("Scrolled Up!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'scroll_down':
                                if scroll_window("down"):
                                    _display_action_status = ("Scrolled Down!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'open_browser':
                                if open_web_browser():
                                    _display_action_status = ("Browser Opened!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'minimize_window':
                                if minimize_active_window():
                                    _display_action_status = ("Window Minimized!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'play_pause':
                                if play_pause_media():
                                    _display_action_status = ("Play/Pause Toggled!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'next_track':
                                if next_track():
                                    _display_action_status = ("Next Track!", current_time)
                                    action_successful = True
                            elif predicted_action_label == 'prev_track':
                                if prev_track():
                                    _display_action_status = ("Previous Track!", current_time)
                                    action_successful = True

                            if action_successful:
                                last_action_time[predicted_action_label] = current_time
                                # Optional: Play a short sound for feedback
                                # try:
                                #     import winsound # Windows only
                                #     winsound.Beep(500, 100)
                                # except ImportError:
                                #     pass # Or use a cross-platform audio library

                else:
                    predicted_action_label = "Waiting for Right Hand"

            # On-screen display (OSD)
            display_text = f'Predicted: {predicted_action_label}'
            if CONFIG['show_confidence']:
                display_text += f' ({prediction_confidence*100:.1f}%)'

            cv2.putText(image, display_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            display_msg, msg_start_time = _display_action_status
            current_time = time.time()
            if display_msg and (current_time - msg_start_time) < CONFIG['display_duration_seconds']:
                cv2.putText(image, display_msg,
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # Clear message if display duration passed
                _display_action_status = ("", 0.0)

            if CONFIG['show_fps']:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

            cv2.imshow('Gesture Control', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print_colored("Q pressed. Exiting real-time control.", COLOR_MAGENTA)
                break

    cap.release()
    cv2.destroyAllWindows()
    print_colored("Program exited.", COLOR_MAGENTA)
    time.sleep(1)

# --- Settings Menu ---
def settings_menu():
    clear_screen()
    print_colored("--- APPLICATION SETTINGS ---", COLOR_YELLOW + ";" + COLOR_BOLD)
    print(f"1. Number of samples per action: {CONFIG['num_samples_per_action']}")
    print(f"2. Prediction Confidence Threshold: {CONFIG['confidence_threshold']:.2f}")
    print(f"3. Gesture Cooldown Seconds: {CONFIG['cooldown_seconds']:.1f}")
    print(f"4. Action Message Display Duration: {CONFIG['display_duration_seconds']:.1f}s")
    print(f"5. Camera Index: {CONFIG['camera_index']} (Available: {', '.join(map(str, get_available_cameras()))})")
    print(f"6. Show FPS in Live Mode: {'Yes' if CONFIG['show_fps'] else 'No'}")
    print(f"7. Show Confidence in Live Mode: {'Yes' if CONFIG['show_confidence'] else 'No'}")
    print(f"8. Show Landmarks in Live Mode: {'Yes' if CONFIG['show_landmarks_in_live_mode'] else 'No'}")
    print("\n[b] Back to Main Menu") # Changed from 'Unsaved changes will be lost' as changes are saved immediately

    while True:
        sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter setting number to change, or 'b' to go back: \033[0m")
        sys.stdout.flush()
        choice = input().strip().lower()

        if choice == 'b':
            save_config() # Save configuration when going back to main menu
            print_colored("Settings saved and returning to main menu.", COLOR_GREEN)
            time.sleep(2)
            return
        elif choice == '1':
            try:
                new_val = int(input(f"Enter new number of samples per action (current: {CONFIG['num_samples_per_action']}): "))
                if new_val > 0:
                    CONFIG['num_samples_per_action'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be positive.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '2':
            try:
                new_val = float(input(f"Enter new confidence threshold (0.0-1.0, current: {CONFIG['confidence_threshold']:.2f}): "))
                if 0.0 <= new_val <= 1.0:
                    CONFIG['confidence_threshold'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be between 0.0 and 1.0.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '3':
            try:
                new_val = float(input(f"Enter new cooldown seconds (current: {CONFIG['cooldown_seconds']:.1f}): "))
                if new_val >= 0:
                    CONFIG['cooldown_seconds'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be non-negative.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '4':
            try:
                new_val = float(input(f"Enter new action message display duration in seconds (current: {CONFIG['display_duration_seconds']:.1f}): "))
                if new_val >= 0.5:
                    CONFIG['display_duration_seconds'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be at least 0.5 seconds.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '5':
            available_cams = get_available_cameras()
            if not available_cams:
                print_colored("No cameras detected.", COLOR_RED)
                time.sleep(1)
                continue
            print(f"Available cameras: {', '.join(map(str, available_cams))}")
            try:
                new_val = int(input(f"Enter new camera index (current: {CONFIG['camera_index']}): "))
                if new_val in available_cams:
                    CONFIG['camera_index'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Invalid camera index.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '6':
            CONFIG['show_fps'] = not CONFIG['show_fps']
            print_colored(f"Show FPS set to: {'Yes' if CONFIG['show_fps'] else 'No'}. Updated.", COLOR_GREEN)
        elif choice == '7':
            CONFIG['show_confidence'] = not CONFIG['show_confidence']
            print_colored(f"Show Confidence set to: {'Yes' if CONFIG['show_confidence'] else 'No'}. Updated.", COLOR_GREEN)
        elif choice == '8':
            CONFIG['show_landmarks_in_live_mode'] = not CONFIG['show_landmarks_in_live_mode']
            print_colored(f"Show Landmarks in Live Mode set to: {'Yes' if CONFIG['show_landmarks_in_live_mode'] else 'No'}. Updated.", COLOR_GREEN)
        else:
            print_colored("Invalid choice.", COLOR_RED)
        time.sleep(0.5) # Short pause to see update before re-prompting
        clear_screen() # Refresh menu display
        print_colored("--- APPLICATION SETTINGS ---", COLOR_YELLOW + ";" + COLOR_BOLD)
        print(f"1. Number of samples per action: {CONFIG['num_samples_per_action']}")
        print(f"2. Prediction Confidence Threshold: {CONFIG['confidence_threshold']:.2f}")
        print(f"3. Gesture Cooldown Seconds: {CONFIG['cooldown_seconds']:.1f}")
        print(f"4. Action Message Display Duration: {CONFIG['display_duration_seconds']:.1f}s")
        print(f"5. Camera Index: {CONFIG['camera_index']} (Available: {', '.join(map(str, get_available_cameras()))})")
        print(f"6. Show FPS in Live Mode: {'Yes' if CONFIG['show_fps'] else 'No'}")
        print(f"7. Show Confidence in Live Mode: {'Yes' if CONFIG['show_confidence'] else 'No'}")
        print(f"8. Show Landmarks in Live Mode: {'Yes' if CONFIG['show_landmarks_in_live_mode'] else 'No'}")
        print("\n[b] Back to Main Menu")

# --- Reset Function ---
def reset_application_data():
    """
    Deletes trained model, all collected gesture data, and resets settings to default.
    Requires user confirmation.
    """
    clear_screen()
    print_colored("--- RESET ALL APPLICATION DATA ---", COLOR_RED + ";" + COLOR_BOLD)
    print_colored("WARNING: This will permanently delete:", COLOR_RED)
    print_colored(f"  - Trained Model: {CONFIG['model_path']}", COLOR_RED)
    print_colored(f"  - All Gesture Data: {CONFIG['data_dir']}", COLOR_RED)
    print_colored(f"  - Active Actions File: {ACTIVE_ACTIONS_FILE}", COLOR_RED)
    print_colored(f"  - Your Settings File: {CONFIG_FILE}", COLOR_RED)
    print_colored("\nThis action CANNOT be undone.", COLOR_RED + ";" + COLOR_BOLD)

    sys.stdout.write(f"\033[{COLOR_YELLOW};{COLOR_BOLD}mAre you sure you want to proceed? (yes/no): \033[0m")
    sys.stdout.flush()
    confirmation = input().strip().lower()

    if confirmation == 'yes':
        # Delete model file
        if os.path.exists(CONFIG['model_path']):
            try:
                os.remove(CONFIG['model_path'])
                print_colored(f"Deleted model: {CONFIG['model_path']}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting model file: {e}", COLOR_RED)
        else:
            print_colored("Model file not found (already deleted or never trained).", COLOR_YELLOW)

        # Delete gesture data directory
        if os.path.exists(CONFIG['data_dir']):
            try:
                shutil.rmtree(CONFIG['data_dir'])
                print_colored(f"Deleted gesture data directory: {CONFIG['data_dir']}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting data directory: {e}", COLOR_RED)
        else:
            print_colored("Gesture data directory not found (already deleted or no data collected).", COLOR_YELLOW)

        # Delete active actions file
        if os.path.exists(ACTIVE_ACTIONS_FILE):
            try:
                os.remove(ACTIVE_ACTIONS_FILE)
                print_colored(f"Deleted active actions file: {ACTIVE_ACTIONS_FILE}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting active actions file: {e}", COLOR_RED)
        else:
            print_colored("Active actions file not found.", COLOR_YELLOW)

        # Reset settings to default by deleting current config and reloading
        if os.path.exists(CONFIG_FILE):
            try:
                os.remove(CONFIG_FILE)
                print_colored(f"Deleted settings file: {CONFIG_FILE}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting settings file: {e}", COLOR_RED)
        else:
            print_colored("Settings file not found.", COLOR_YELLOW)

        print_colored("\nApplication data has been reset to default.", COLOR_GREEN + ";" + COLOR_BOLD)
        print_colored("Configuration will be reloaded.", COLOR_YELLOW)
        time.sleep(3) # Give user time to read success message
    else:
        print_colored("Reset cancelled.", COLOR_CYAN)
        time.sleep(1.5)

# --- Main Application Logic ---
def main():
    """Main function to run the gesture control application."""
    load_config() # Load configuration at startup

    choice = ''
    while choice != 'e':
        clear_screen()
        print_colored("==============================================", COLOR_BLUE + ";" + COLOR_BOLD)
        print_colored("     WELCOME TO GESTURE CONTROL APPLICATION!    ", COLOR_CYAN + ";" + COLOR_BOLD)
        print_colored("==============================================", COLOR_BLUE + ";" + COLOR_BOLD)
        print_colored("\n[n] Add New Gestures & Train Model", COLOR_GREEN)
        print_colored("[a] Activate Real-time Gesture Control", COLOR_YELLOW)
        print_colored("[s] Settings", COLOR_MAGENTA)
        print_colored("[r] Reset All Data (Caution: Deletes all trained models and settings!)", COLOR_RED) # New option
        print_colored("[e] Exit Application", COLOR_RED)
        print_colored("----------------------------------------------", COLOR_BLUE)

        sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter your choice (n/a/s/r/e): \033[0m")
        sys.stdout.flush()
        choice = input().strip().lower()

        if choice == 'n':
            start_gathering()
            train_model()
        elif choice == 'a':
            activate_gesture_control()
        elif choice == 's': # Handle settings menu
            settings_menu()
        elif choice == 'r': # Handle reset
            reset_application_data()
            load_config() # Reload config after reset to ensure current session uses default values
        elif choice == 'e':
            print_colored("\nExiting application. Goodbye!", COLOR_MAGENTA)
            time.sleep(1)
            break
        else:
            print_colored("Invalid choice. Please enter 'n', 'a', 's', 'r', or 'e'.", COLOR_RED)
            time.sleep(1.5)

    print_colored("Application closed.", COLOR_MAGENTA + ";" + COLOR_BOLD)

if __name__ == "__main__":
    main()