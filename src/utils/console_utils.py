# src/utils/console_utils.py
import os
import sys
import time

from src.constants import COLOR_RESET, COLOR_BLUE, COLOR_BOLD, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_CYAN

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color_code):
    """Prints text in a specified ANSI color."""
    sys.stdout.write(f"\033[{color_code}m{text}\033[0m\n")
    sys.stdout.flush()

def progress_bar(current, total, bar_length=50, title="Progress"):
    """
    Displays a terminal-based progress bar.

    Args:
        current (int): The current progress value.
        total (int): The total value for completion.
        bar_length (int): The length of the progress bar in characters.
        title (str): A title for the progress bar.
    """
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r{title}: [{arrow}{spaces}] {int(percent * 100)}% ({current}/{total})')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()