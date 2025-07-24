# src/system_controls/volume.py
import pyautogui
import sys
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

_has_pyautogui = False
try:
    import pyautogui
    _has_pyautogui = True
except ImportError:
    print_colored("Warning: 'pyautogui' not found. Volume control and other UI interactions will not work.", COLOR_YELLOW)
except Exception as e:
    print_colored(f"Warning: Could not initialize 'pyautogui': {e}. UI control may not work.", COLOR_YELLOW)


def adjust_volume(direction):
    """
    Adjusts system volume up or down.
    Args:
        direction (str): 'up' or 'down'.
    Returns:
        bool: True if successful, False otherwise.
    """
    if not _has_pyautogui:
        print_colored("Volume control is not available (pyautogui module not loaded).", COLOR_RED)
        return False

    try:
        if direction == "up":
            pyautogui.press('volumeup')
            return True
        elif direction == "down":
            pyautogui.press('volumedown')
            return True
        else:
            print_colored(f"Invalid volume direction: {direction}", COLOR_RED)
            return False
    except Exception as e:
        print_colored(f"Error adjusting volume: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False