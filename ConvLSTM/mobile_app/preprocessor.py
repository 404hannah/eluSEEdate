"""
Video Preprocessor for Mobile ConvLSTM Turn Prediction

Based on Prototype 9 preprocessing pipeline
Prepares video frames for inference on Android devices using Expo

Key Features:
- Extracts frames at 10 FPS from video
- Resizes to 128x128
- Normalizes pixel values to [0, 1]
- Adds intent channels (3 additional channels)
- Returns tensor shape: [seq_len, channels, height, width] = [20, 6, 128, 128]
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

from config import CONFIG, HEIGHT, WIDTH, CHANNELS, FPS, SEQ_LEN

class VideoPreprocessor:
    """
    Preprocesses video input for ConvLSTM model inference
    
    Compatible with Prototype 9 model architecture
    Designed for real-time processing on mobile devices
    """
    
    def __init__(
        self, 
        height: int = HEIGHT,
        width: int = WIDTH,
        fps: int = FPS,
        seq_len: int = SEQ_LEN,
        normalize: bool = True
    ):
        """
        Initialize the video preprocessor
        
        Args:
            height: Target frame height (default: 128)
            width: Target frame width (default: 128)
            fps: Frames per second to extract (default: 10)
            seq_len: Number of frames to extract (default: 20)
            normalize: Whether to normalize pixel values to [0, 1] (default: True)
        """
        self.height = height
        self.width = width
        self.fps = fps
        self.seq_len = seq_len
        self.normalize = normalize
        
    def preprocess_video(
        self,
        video_path: Union[str, Path],
        intent: Optional[int] = None,
        intent_position: Optional[int] = None
    ) -> np.ndarray:
        """
        Preprocess a video file for model inference
        
        Pipeline:
        1. Extract frames at specified FPS
        2. Resize to (height, width)
        3. Convert BGR to RGB
        4. Normalize to [0, 1] if enabled
        5. Add intent channels
        6. Stack into sequence
        
        Args:
            video_path: Path to video file
            intent: Direction intent (0: Front, 1: Left, 2: Right, None: no intent)
            intent_position: Frame index where intent starts (defaults to 0 if None)
                            For mobile apps, always pass explicit position from direction detection
        
        Returns:
            np.ndarray: Preprocessed video tensor of shape [seq_len, channels, height, width]
                       [20, 6, 128, 128] by default
        
        Example:
            >>> preprocessor = VideoPreprocessor()
            >>> video_tensor = preprocessor.preprocess_video("sample.mp4", intent=1)
            >>> print(video_tensor.shape)  # (20, 6, 128, 128)
        """
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(original_fps / self.fps))
        
        # Use default intent position if not provided
        # For mobile deployment: always pass explicit intent_position from direction detection
        if intent is not None and intent_position is None:
            intent_position = 0  # Default to start of video (testing/fallback only)
        
        frames = []
        frame_idx = 0
        extracted_count = 0
        
        while extracted_count < self.seq_len:
            ret, frame = cap.read()
            
            if not ret:
                # If video ends early, pad with zeros
                zero_frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
                frame_tensor = self._process_frame(
                    zero_frame, 
                    extracted_count, 
                    intent, 
                    intent_position
                )
                frames.append(frame_tensor)
                extracted_count += 1
                continue
            
            # Sample frames at desired FPS
            if frame_idx % frame_interval == 0:
                # Process frame
                frame_tensor = self._process_frame(
                    frame,
                    extracted_count,
                    intent,
                    intent_position
                )
                frames.append(frame_tensor)
                extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Stack frames into sequence
        video_tensor = np.stack(frames, axis=0)  # [seq_len, channels, height, width]
        
        return video_tensor
    
    # LIVE CAMERA FEED PROCESSING
    def preprocess_frame_sequence(
        self,
        frames: list,
        intent: Optional[int] = None,
        intent_position: Optional[int] = None
    ) -> np.ndarray:
        """
        Preprocess a sequence of frames (already extracted)
        
        For real-time camera feed processing
        
        Args:
            frames: List of numpy arrays (BGR format)
            intent: Direction intent (0: Front, 1: Left, 2: Right, None: no intent)
            intent_position: Frame index where intent starts (defaults to 0 if None)
                            For mobile apps, always pass explicit position from direction detection
        
        Returns:
            np.ndarray: Preprocessed video tensor of shape [seq_len, channels, height, width]
        """
        if len(frames) != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} frames, got {len(frames)}")
        
        # Use default intent position if not provided
        # For mobile deployment: always pass explicit intent_position from direction detection
        if intent is not None and intent_position is None:
            intent_position = 0  # Default to start of video (testing/fallback only)
        
        processed_frames = []
        for i, frame in enumerate(frames):
            frame_tensor = self._process_frame(frame, i, intent, intent_position)
            processed_frames.append(frame_tensor)
        
        video_tensor = np.stack(processed_frames, axis=0)
        return video_tensor
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intent: Optional[int],
        intent_position: Optional[int]
    ) -> np.ndarray:
        """
        Process a single frame through the preprocessing pipeline
        
        Steps:
        1. Resize to target dimensions
        2. Convert BGR to RGB
        3. Normalize to [0, 1]
        4. Transpose to channels-first format
        5. Add intent channels
        
        Args:
            frame: Input frame (BGR format)
            frame_idx: Index of frame in sequence
            intent: Direction intent
            intent_position: Frame index where intent starts
        
        Returns:
            np.ndarray: Processed frame of shape [channels, height, width]
        """
        # Resize frame
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)
        
        # Transpose to channels-first: [H, W, C] -> [C, H, W]
        frame = np.transpose(frame, (2, 0, 1))
        
        # Add intent channels
        frame_with_intent = self._add_intent_channels(
            frame, frame_idx, intent, intent_position
        )
        
        return frame_with_intent
    
    def _add_intent_channels(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intent: Optional[int],
        intent_position: Optional[int]
    ) -> np.ndarray:
        """
        Add 3 intent channels to the frame
        
        Intent Encoding:
        - No intent: All 3 channels are zeros
        - Intent exists: One channel (corresponding to direction) is filled with 1s
                        for intent_frames (10 frames = 1 second)
        
        Args:
            frame: Frame tensor of shape [3, height, width]
            frame_idx: Current frame index in sequence
            intent: Direction intent (0: Front, 1: Left, 2: Right, None: no intent)
            intent_position: Frame index where intent signal starts
        
        Returns:
            np.ndarray: Frame with intent channels [6, height, width]
        """
        # Initialize intent channels (all zeros)
        intent_channels = np.zeros((3, self.height, self.width), dtype=np.float32)
        
        # If intent exists and frame is within intent duration
        if intent is not None and intent_position is not None:
            intent_duration = CONFIG.intent_frames  # 10 frames (1 second)
            
            # Check if current frame should have intent signal
            if intent_position <= frame_idx < (intent_position + intent_duration):
                # Set the appropriate intent channel to 1
                intent_channels[intent, :, :] = 1.0
        
        # Concatenate RGB channels with intent channels
        frame_with_intent = np.concatenate([frame, intent_channels], axis=0)
        
        return frame_with_intent
    
    def get_output_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the expected output shape after preprocessing
        
        Returns:
            Tuple[int, int, int, int]: (seq_len, channels, height, width)
        """
        return (self.seq_len, CHANNELS, self.height, self.width)

def preprocess_for_inference(
    video_path: Union[str, Path],
    intent: Optional[int] = None,
    intent_position: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function for quick preprocessing
    
    Args:
        video_path: Path to video file
        intent: Direction intent (0: Front, 1: Left, 2: Right, None: no intent)
        intent_position: Frame index where intent starts (defaults to 0 if None)
                        For mobile apps, always pass explicit position from direction detection
    
    Returns:
        np.ndarray: Preprocessed video tensor ready for model inference
                   Shape: [1, seq_len, channels, height, width] (batch dimension added)
    
    Example:
        >>> video_tensor = preprocess_for_inference("sample.mp4", intent=1)
        >>> # Ready to feed to model
        >>> prediction = model(video_tensor)
    """
    preprocessor = VideoPreprocessor()
    video_tensor = preprocessor.preprocess_video(video_path, intent, intent_position)
    
    # Add batch dimension: [seq_len, C, H, W] -> [1, seq_len, C, H, W]
    video_tensor = np.expand_dims(video_tensor, axis=0)
    
    return video_tensor

if __name__ == "__main__":
    # Example usage
    print("Video Preprocessor for Mobile ConvLSTM")
    print(f"Expected output shape: {VideoPreprocessor().get_output_shape()}")
    print(f"Configuration: {HEIGHT}x{WIDTH}, {FPS} FPS, {SEQ_LEN} frames")
    print(f"Channels: {CHANNELS} (3 RGB + 3 intent)")