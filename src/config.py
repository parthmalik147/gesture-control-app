# src/config.py
import json
import os
from src.constants import DEFAULT_CONFIG, CONFIG_FILE_PATH, APP_DATA_BASE_DIR
from src.utils.console_utils import print_colored, COLOR_GREEN, COLOR_RED, COLOR_YELLOW

# Global dictionary to hold the application configuration
CONFIG = {}

def load_config():
    """
    Loads configuration from CONFIG_FILE_PATH.
    If the file doesn't exist or is invalid, it initializes with DEFAULT_CONFIG
    and saves it.
    """
    global CONFIG
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                loaded_config = json.load(f)
                # Merge loaded config with default, prioritizing loaded values
                # This ensures new default keys are added if config file is old
                CONFIG = {**DEFAULT_CONFIG, **loaded_config}
            print_colored(f"Configuration loaded from {CONFIG_FILE_PATH}", COLOR_GREEN)
        except json.JSONDecodeError:
            print_colored(f"Error reading config file. Using default settings.", COLOR_RED)
            CONFIG = DEFAULT_CONFIG.copy()
        except Exception as e:
            print_colored(f"An unexpected error occurred loading config: {e}. Using default settings.", COLOR_RED)
            CONFIG = DEFAULT_CONFIG.copy()
    else:
        CONFIG = DEFAULT_CONFIG.copy()
        save_config() # Save the default config if file doesn't exist
        print_colored(f"No config file found. Created default config at {CONFIG_FILE_PATH}", COLOR_YELLOW)

    # Ensure data directories are set relative to APP_DATA_BASE_DIR
    # These paths are derived and not meant to be directly in app_config.json
    CONFIG['data_dir'] = os.path.join(APP_DATA_BASE_DIR, 'simple_gesture_data')
    CONFIG['model_path'] = os.path.join(APP_DATA_BASE_DIR, 'trained_models', 'simple_gesture_model.h5')

    # Ensure necessary directories exist
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    os.makedirs(os.path.join(APP_DATA_BASE_DIR, 'trained_models'), exist_ok=True)


def save_config():
    """
    Saves the current CONFIG to CONFIG_FILE_PATH.
    Filters out transient keys like 'data_dir' and 'model_path' that are derived.
    """
    try:
        # Create a copy and remove keys that should not be persisted in the JSON file
        config_to_save = {k: v for k, v in CONFIG.items() if k not in ['data_dir', 'model_path']}
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print_colored(f"Configuration saved to {CONFIG_FILE_PATH}", COLOR_GREEN)
    except Exception as e:
        print_colored(f"Error saving configuration: {e}", COLOR_RED)

def get_config():
    """
    Returns the current application configuration.
    """
    return CONFIG

# Load configuration when the module is imported for the first time
# This ensures CONFIG is populated before other modules try to access it
if not CONFIG: # Only load if it's empty, preventing re-loading issues
    load_config()