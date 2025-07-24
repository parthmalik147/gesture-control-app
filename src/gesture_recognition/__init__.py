# src/gesture_recognition/__init__.py
# This file makes the 'gesture_recognition' directory a Python package.

# Import key functions from recognizer and action_handler
from .recognizer import activate_gesture_control
from .action_handler import execute_gesture_action

__all__ = [
    "activate_gesture_control",
    "execute_gesture_action"
]