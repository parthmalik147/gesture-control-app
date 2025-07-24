# src/system_controls/__init__.py
# This file makes the 'system_controls' directory a Python package.

# Import all public functions from the system control modules
from .brightness import adjust_brightness
from .volume import adjust_volume
from .scroll import scroll_window
from .media import play_pause_media, next_track, prev_track
from .window_management import open_web_browser, minimize_active_window

__all__ = [
    "adjust_brightness",
    "adjust_volume",
    "scroll_window",
    "play_pause_media",
    "next_track",
    "prev_track",
    "open_web_browser",
    "minimize_active_window",
]