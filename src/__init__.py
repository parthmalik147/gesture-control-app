# src/__init__.py
# This file makes the 'src' directory a Python package.

# Expose core components directly under 'src' for cleaner imports
from . import config
from . import constants
from . import main # Although main is usually run directly, it can be imported
from .utils import camera_utils, console_utils # Expose key utilities directly under src.utils
from .data_collection import collector
from .model_training import trainer
from .gesture_recognition import recognizer, action_handler
from .system_controls import brightness, volume, scroll, media, window_management

# You can also define an __all__ variable to explicitly list what's part of the public API
__all__ = [
    "config",
    "constants",
    "main",
    "camera_utils",
    "console_utils",
    "collector",
    "trainer",
    "recognizer",
    "action_handler",
    "brightness",
    "volume",
    "scroll",
    "media",
    "window_management"
]