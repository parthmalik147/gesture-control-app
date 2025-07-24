# src/model_training/__init__.py
# This file makes the 'model_training' directory a Python package.

# Import the main function from trainer to be directly accessible
from .trainer import train_model

__all__ = [
    "train_model"
]