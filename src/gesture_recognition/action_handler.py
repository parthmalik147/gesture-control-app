# src/gesture_recognition/action_handler.py
import time
from src.system_controls.brightness import adjust_brightness
from src.system_controls.volume import adjust_volume
from src.system_controls.scroll import scroll_window
from src.system_controls.media import play_pause_media, next_track, prev_track
from src.system_controls.window_management import open_web_browser, minimize_active_window #, open_calculator_windows
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

# Keep track of last action time globally within this module
_last_action_time = {}

def execute_gesture_action(action_label, config, current_brightness_val):
    """
    Executes the system action corresponding to the recognized gesture label.
    Applies a cooldown period to prevent rapid, unintended actions.

    Args:
        action_label (str): The label of the recognized gesture.
        config (dict): The application configuration dictionary.
        current_brightness_val (int): The current brightness value, needed for brightness adjustments.

    Returns:
        tuple: (display_message, action_time, updated_brightness_value)
               display_message (str or None): Message to display on screen, or None if no action.
               action_time (float or None): Timestamp of when the action was executed.
               updated_brightness_value (int): The new brightness value after adjustment.
    """
    global _last_action_time
    current_time = time.time()
    display_message = None
    action_successful = False
    updated_brightness = current_brightness_val # Assume no change unless explicitly adjusted

    # Check cooldown for the specific action
    if (current_time - _last_action_time.get(action_label, 0)) > config['cooldown_seconds']:
        if action_label == 'brightness_up':
            updated_brightness = adjust_brightness(10, current_brightness_val)
            display_message = f"Brightness: {updated_brightness}%"
            action_successful = True
        elif action_label == 'brightness_down':
            updated_brightness = adjust_brightness(-10, current_brightness_val)
            display_message = f"Brightness: {updated_brightness}%"
            action_successful = True
        elif action_label == 'volume_up':
            if adjust_volume("up"):
                display_message = "Volume Up!"
                action_successful = True
        elif action_label == 'volume_down':
            if adjust_volume("down"):
                display_message = "Volume Down!"
                action_successful = True
        elif action_label == 'scroll_up':
            if scroll_window("up"):
                display_message = "Scrolled Up!"
                action_successful = True
        elif action_label == 'scroll_down':
            if scroll_window("down"):
                display_message = "Scrolled Down!"
                action_successful = True
        elif action_label == 'open_browser':
            if open_web_browser():
                display_message = "Browser Opened!"
                action_successful = True
        elif action_label == 'minimize_window':
            if minimize_active_window():
                display_message = "Window Minimized!"
                action_successful = True
        elif action_label == 'play_pause':
            if play_pause_media():
                display_message = "Play/Pause Toggled!"
                action_successful = True
        elif action_label == 'next_track':
            if next_track():
                display_message = "Next Track!"
                action_successful = True
        elif action_label == 'prev_track':
            if prev_track():
                display_message = "Previous Track!"
                action_successful = True
        elif action_label == 'neutral':
            # Neutral gesture does nothing, but can reset cooldown if needed for other gestures
            pass
        else:
            print_colored(f"Unhandled action: {action_label}", COLOR_RED)
            print_colored("Ensure this action is defined in 'actions' in config/app_config.json and has a corresponding handler.", COLOR_YELLOW)


        if action_successful:
            _last_action_time[action_label] = current_time # Update last action time only if action was successful
            return display_message, current_time, updated_brightness
    return None, None, current_brightness_val # No action taken (due to cooldown or unhandled action)