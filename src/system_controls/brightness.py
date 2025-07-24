# src/system_controls/brightness.py
import sys
import warnings
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

_has_sbc = False
try:
    import screen_brightness_control as sbc
    _has_sbc = True
except ImportError:
    print_colored("Warning: 'screen_brightness_control' not found. Brightness control will not work.", COLOR_YELLOW)
except Exception as e:
    print_colored(f"Warning: Could not initialize 'screen_brightness_control': {e}. Brightness control may not work.", COLOR_YELLOW)

def adjust_brightness(delta, current_brightness_val):
    """
    Adjusts screen brightness by a given delta.
    Returns the new brightness percentage or current_brightness_val if adjustment fails.
    """
    if not _has_sbc:
        print_colored("Brightness control is not available (screen_brightness_control module not loaded).", COLOR_RED)
        return current_brightness_val

    try:
        current_brightness_list = sbc.get_brightness()
        # Assume the first monitor's brightness for simplicity or find a primary display
        if current_brightness_list:
            current_brightness = current_brightness_list[0]
        else:
            print_colored("Could not get current brightness. Using default 50% for calculation.", COLOR_YELLOW)
            current_brightness = 50 # Default if no brightness can be read

        new_brightness = max(0, min(100, current_brightness + delta))
        sbc.set_brightness(new_brightness)
        return new_brightness
    except sbc.exceptions.ScreenBrightnessError as e:
        print_colored(f"Error adjusting brightness: {e}. Check permissions or display configuration.", COLOR_RED)
    except Exception as e:
        print_colored(f"An unexpected error occurred during brightness control: {e}", COLOR_RED)
    return current_brightness_val