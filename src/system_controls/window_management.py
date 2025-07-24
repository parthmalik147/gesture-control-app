# src/system_controls/window_management.py
import webbrowser
import pyautogui
import sys
import subprocess
from src.utils.console_utils import print_colored, COLOR_RED, COLOR_YELLOW

_has_pyautogui = False
try:
    import pyautogui
    _has_pyautogui = True
except ImportError:
    print_colored("Warning: 'pyautogui' not found. Window management may be limited.", COLOR_YELLOW)
except Exception as e:
    print_colored(f"Warning: Could not initialize 'pyautogui': {e}. Window management may be limited.", COLOR_YELLOW)


def open_web_browser(url="https://www.google.com"):
    """Opens the default web browser to a given URL."""
    try:
        webbrowser.open(url)
        return True
    except Exception as e:
        print_colored(f"Error opening web browser: {e}", COLOR_RED)
        return False

def minimize_active_window():
    """Minimizes the currently active window."""
    if not _has_pyautogui:
        print_colored("Window minimization is not available (pyautogui module not loaded).", COLOR_RED)
        return False
    try:
        if sys.platform == "win32":
            pyautogui.hotkey('win', 'down') # Windows: Minimize active window
        elif sys.platform == "darwin":
            # macOS: Hide current application. Minimizing specific window is more complex.
            # This simulates Cmd+H
            pyautogui.hotkey('command', 'h')
        elif sys.platform.startswith("linux"):
            # Linux (e.g., Ubuntu with Gnome): Minimize active window
            # This might vary between desktop environments.
            # XF86Launch5 is a common key for minimize on some systems
            # Alt+Space then N for minimize in some setups
            pyautogui.hotkey('alt', 'space')
            pyautogui.press('n')
        else:
            print_colored("Minimize active window is not supported on this OS.", COLOR_YELLOW)
            return False
        return True
    except Exception as e:
        print_colored(f"Error minimizing window: {e}", COLOR_RED)
        if sys.platform == "darwin":
            print_colored("On macOS, you might need to grant Accessibility permissions to your terminal or IDE.", COLOR_YELLOW)
        return False

# You had some subprocess calls for other actions, which can be placed here if needed.
# For example, opening calculator on Windows:
def open_calculator_windows():
    if sys.platform == "win32":
        try:
            subprocess.Popen('calc.exe')
            return True
        except Exception as e:
            print_colored(f"Error opening calculator: {e}", COLOR_RED)
            return False
    else:
        print_colored("Calculator action is only for Windows.", COLOR_YELLOW)
        return False