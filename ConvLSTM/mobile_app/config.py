"""
Mobile App Configuration for ConvLSTM Turn Prediction Model

Based on Prototype 9
Configuration constants for preprocessing video input on Android devices
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MobileConfig:
    """Configuration for mobile preprocessing and inference."""
    
    # Video Input Specifications
    height: int = 128                # Frame height after resize
    width: int = 128                 # Frame width after resize
    channels: int = 6                # 3 (RGB) + 3 (intent channels)
    fps: int = 10                    # Frames per second to extract
    duration: int = 2                # Video duration in seconds
    seq_len: int = 20                # Total frames (fps * duration)
    
    # Model Output
    num_classes: int = 3             # 0: Front, 1: Left, 2: Right
    class_names: List[str] = None    # Human-readable class labels
    
    # Preprocessing
    normalize: bool = True           # Normalize pixel values to [0, 1]
    color_format: str = "RGB"        # Expected color format (RGB)
    
    # Intent Settings
    intent_duration: int = 1         # Intent signal duration in seconds
    intent_frames: int = 10          # Intent frames (intent_duration * fps)
    intent_probability: float = 0.6  # Probability of intent being present
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Front", "Left", "Right"]
        self.seq_len = self.fps * self.duration
        self.intent_frames = self.intent_duration * self.fps

# Global config instance
CONFIG = MobileConfig()

# Convenience exports
HEIGHT = CONFIG.height
WIDTH = CONFIG.width
CHANNELS = CONFIG.channels
FPS = CONFIG.fps
DURATION = CONFIG.duration
SEQ_LEN = CONFIG.seq_len
NUM_CLASSES = CONFIG.num_classes
CLASS_NAMES = CONFIG.class_names