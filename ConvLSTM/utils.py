import os
from typing import List, Tuple
from dataclasses import dataclass

# All hyperparameters in one place

@dataclass
class Config:
    """All hyperparameters for Prototype 7."""
    # Data
    height: int = 128
    width: int = 128
    channels: int = 3
    fps: int = 10
    duration: int = 3
    seq_len: int = 30  # fps * duration

    # Training
    batch_size: int = 6
    accumulation_steps: int = 4  # Effective batch = 6 * 4 = 24
    num_epochs: int = 3
    learning_rate: float = 1e-4
    seed: int = 8

    # Early Stopping
    early_stop_patience: int = 5
    min_delta: float = 0.01  # 0.01% improvement threshold

    # Regularization
    dropout_rate: float = 0.5
    max_grad_norm: float = 1.0

    # LR Scheduler
    lr_factor: float = 0.5
    lr_patience: int = 3
    lr_min: float = 1e-7

    # Model Architecture
    input_dim: int = 3
    hidden_dim: List[int] = None  # [64, 32]
    kernel_size: Tuple[int, int] = (3, 3)
    num_layers: int = 2
    num_classes: int = 3

    # Cache
    reserve_gb: float = 10.0
    eviction_check_interval: int = 10
    eviction_buffer_percent: float = 0.10

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = [64, 32]
        self.seq_len = self.fps * self.duration

# Global config instance
CONFIG = Config()

# Convenience aliases for backward compatibility
HEIGHT = CONFIG.height
WIDTH = CONFIG.width
CHANNELS = CONFIG.channels
FPS = CONFIG.fps
DURATION = CONFIG.duration
SEQ_LEN = CONFIG.seq_len

# MVO Prediction Logic Mapping
# FRONT: 0, LEFT: 1, RIGHT: 2
def get_label_id(label_name: str) -> int:
    """Map label name to numeric ID."""
    mapping = {'front': 0, 'left': 1, 'right': 2}
    return mapping.get(label_name.lower(), 0)

# Data Paths

VIDEO_DIR = '/content/drive/MyDrive/Dataset_750/videos/'
LABEL_DIR = '/content/drive/MyDrive/Dataset_750/labels/'

# Cache Directory
CACHE_DIR = '/content/cache/'  # Local SSD (fast, cleared on runtime disconnect)

# Validate paths exist
for path, name in [(VIDEO_DIR, "VIDEO_DIR"), (LABEL_DIR, "LABEL_DIR")]:
    if not os.path.exists(path):
        print(f"WARNING: {name} not found: {path}")