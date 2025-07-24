# src/system_controls/scroll.py
import pyautogui
import sys
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

_has_pyautogui = False
try:
    import pyautogui
    _has_pyautogui = True
except ImportError:
    print_colored("Warning: 'pyautogui' not found. Scrolling will not work.", COLOR_YELLOW)
except Exception as e:
    print_colored(f"Warning: Could not initialize 'pyautogui': {e}. Scrolling may not work.", COLOR_YELLOW)

def scroll_window(direction, amount=100):
    """
    Scrolls the active window up or down.
    Args:
        direction (str): 'up' or 'down'.
        amount (int): The amount to scroll (in pixels).
    Returns:
        bool: True if successful, False otherwise.
    """
    if not _has_pyautogui:
        print_colored("Scrolling is not available (pyautogui module not loaded).", COLOR_RED)
        return False

    try:
        if direction == "up":
            pyautogui.scroll(amount)
            return True
        elif direction == "down":
            pyautogui.scroll(-amount)
            return True
        else:
            print_colored(f"Invalid scroll direction: {direction}", COLOR_RED)
            return False
    except Exception as e:
        print_colored(f"Error scrolling: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False