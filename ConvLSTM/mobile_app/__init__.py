"""
Mobile App Module for ConvLSTM Turn Prediction

Provides preprocessing utilities for deploying the ConvLSTM model
on Android devices using Expo/React Native

Based on Prototype 9 architecture
"""

from .config import CONFIG, MobileConfig
from .preprocessor import VideoPreprocessor, preprocess_for_inference

__version__ = "1.0.0"
__description__ = "Mobile preprocessing for ConvLSTM turn prediction"

__all__ = [
    "CONFIG",
    "MobileConfig",
    "VideoPreprocessor",
    "preprocess_for_inference"
]