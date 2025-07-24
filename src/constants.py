# src/constants.py
import os
import sys

# Determine Application Data Paths
# If bundled (frozen), use a user-specific data directory.
# Otherwise, use the 'data' directory relative to the project root.
if getattr(sys, 'frozen', False):
    APP_DATA_BASE_DIR = os.path.join(os.path.expanduser('~'), '.gesture_control_app_data')
else:
    # This path assumes 'constants.py' is in 'src/' and 'data/' is a sibling of 'src/'.
    APP_DATA_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Configuration File Path (relative to project root/config directory)
# This path assumes 'constants.py' is in 'src/' and 'config/' is a sibling of 'src/'.
CONFIG_FILE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')), 'app_config.json')

# Path for the file storing active actions (used by the model)
ACTIVE_ACTIONS_FILE_PATH = os.path.join(APP_DATA_BASE_DIR, 'trained_models', 'active_actions.json')


# Default Configuration for the application
DEFAULT_CONFIG = {
    'actions': ['brightness_up', 'brightness_down', 'neutral',
                'volume_up', 'volume_down', 'scroll_up', 'scroll_down',
                'open_browser', 'minimize_window',
                'play_pause', 'next_track', 'prev_track'],
    'num_samples_per_action': 300,
    'confidence_threshold': 0.98,
    'cooldown_seconds': 0.5,
    'display_duration_seconds': 2.0,
    'camera_index': 0,
    'show_fps': True,
    'show_confidence': True,
    'show_landmarks_in_live_mode': True
}

# ANSI color codes for console output
COLOR_BLUE = "94"
COLOR_CYAN = "96"
COLOR_GREEN = "92"
COLOR_YELLOW = "93"
COLOR_RED = "91"
COLOR_MAGENTA = "95"
COLOR_WHITE = "97"
COLOR_BOLD = "1"
COLOR_RESET = "0"