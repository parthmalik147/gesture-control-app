# src/system_controls/media.py
import pyautogui
import sys
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

_has_pyautogui = False
try:
    import pyautogui
    _has_pyautogui = True
except ImportError:
    print_colored("Warning: 'pyautogui' not found. Media controls will not work.", COLOR_YELLOW)
except Exception as e:
    print_colored(f"Warning: Could not initialize 'pyautogui': {e}. Media control may not work.", COLOR_YELLOW)


def play_pause_media():
    """Toggles media play/pause."""
    if not _has_pyautogui:
        print_colored("Media control is not available (pyautogui module not loaded).", COLOR_RED)
        return False
    try:
        pyautogui.press('playpause')
        return True
    except Exception as e:
        print_colored(f"Error toggling play/pause: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False

def next_track():
    """Skips to the next media track."""
    if not _has_pyautogui:
        print_colored("Media control is not available (pyautogui module not loaded).", COLOR_RED)
        return False
    try:
        pyautogui.press('nexttrack')
        return True
    except Exception as e:
        print_colored(f"Error skipping to next track: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False

def prev_track():
    """Goes to the previous media track."""
    if not _has_pyautogui:
        print_colored("Media control is not available (pyautogui module not loaded).", COLOR_RED)
        return False
    try:
        pyautogui.press('prevtrack')
        return True
    except Exception as e:
        print_colored(f"Error skipping to previous track: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False