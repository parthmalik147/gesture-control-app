# src/data_collection/__init__.py
# This file makes the 'data_collection' directory a Python package.

# Import the main function from collector to be directly accessible
from .collector import start_gathering

__all__ = [
    "start_gathering"
]