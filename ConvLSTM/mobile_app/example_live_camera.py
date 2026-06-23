"""
Example: Processing Live Camera Feeds
======================================

This example demonstrates how to use the VideoPreprocessor to process
live camera feeds for real-time ConvLSTM turn prediction.

Use Case:
- Real-time predictions in mobile apps
- Expo/React Native integration
- Live camera feed processing
- In-app turn prediction

Date: February 25, 2026
"""

import numpy as np
import cv2
import torch
import time
from collections import deque
from typing import Optional

# Import from mobile_app module
from preprocessor import VideoPreprocessor
from config import CONFIG

class LiveCameraProcessor:
    """
    Wrapper class for processing live camera feeds
    
    Handles frame buffering, intent tracking, and preprocessing
    for real-time turn prediction in mobile apps
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize live camera processor
        
        Args:
            camera_id: Camera device ID (0 for default camera)
        """
        self.preprocessor = VideoPreprocessor()
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Frame buffer (holds last 20 frames)
        self.frame_buffer = deque(maxlen=CONFIG.seq_len)
        
        # Intent tracking
        self.current_intent = None
        self.intent_frame_index = None
        
        # Frame counter
        self.frame_count = 0
        
        print(f"✓ Camera {camera_id} opened successfully")
        print(f"✓ Buffer size: {CONFIG.seq_len} frames")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera
        
        Returns:
            np.ndarray or None: Captured frame in BGR format
        """
        ret, frame = self.camera.read()
        if ret:
            self.frame_buffer.append(frame)
            self.frame_count += 1
            return frame
        return None
    
    def set_intent(self, intent: int):
        """
        Set detected intent (when direction is determined by the model)
        
        Args:
            intent: Direction (0=Front, 1=Left, 2=Right)
        """
        self.current_intent = intent
        # Record the frame index within current buffer
        self.intent_frame_index = len(self.frame_buffer) - 1
        
        intent_name = CONFIG.class_names[intent]
        print(f"✓ Intent set: {intent_name} at frame {self.frame_count}")
    
    def clear_intent(self):
        """Clear current intent signal"""
        self.current_intent = None
        self.intent_frame_index = None
        print("✓ Intent cleared")
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames for preprocessing"""
        return len(self.frame_buffer) == CONFIG.seq_len
    
    def preprocess_current_buffer(self) -> np.ndarray:
        """
        Preprocess frames currently in buffer
        
        Returns:
            np.ndarray: Preprocessed tensor [20, 6, 128, 128]
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready. Have {len(self.frame_buffer)}/{CONFIG.seq_len} frames"
            )
        
        # Convert deque to list
        frames = list(self.frame_buffer)
        
        # Preprocess frame sequence
        video_tensor = self.preprocessor.preprocess_frame_sequence(
            frames=frames,
            intent=self.current_intent,
            intent_position=self.intent_frame_index
        )
        
        return video_tensor
    
    def release(self):
        """Release camera resource"""
        self.camera.release()
        print("✓ Camera released")

def example_basic_live_capture():
    """
    Example 1: Basic live camera capture and preprocessing
    
    Capture 20 frames from camera and preprocess them
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Live Camera Capture")
    print("=" * 70)
    
    # Initialize camera processor
    processor = LiveCameraProcessor(camera_id=0)
    
    print("Capturing 20 frames from camera...")
    
    # Capture frames
    for i in range(CONFIG.seq_len):
        frame = processor.capture_frame()
        if frame is not None:
            print(f"  Frame {i+1}/{CONFIG.seq_len} captured", end='\r')
        else:
            print(f"\n✗ Failed to capture frame {i+1}")
            processor.release()
            return
    
    print(f"\n✓ All {CONFIG.seq_len} frames captured")
    
    # Set intent (simulate direction detection at frame 8)
    processor.intent_frame_index = 8
    processor.current_intent = 1  # Left turn
    
    # Preprocess
    video_tensor = processor.preprocess_current_buffer()
    
    print(f"✓ Preprocessing complete")
    print(f"  Output shape: {video_tensor.shape}")  # (20, 6, 128, 128)
    print(f"  Intent: Left turn at frame 8")
    
    # Release camera
    processor.release()
    print()

def example_continuous_capture():
    """
    Example 2: Continuous capture with rolling buffer
    
    Simulates real-time processing where we continuously capture frames
    and can trigger preprocessing at any time
    """
    print("=" * 70)
    print("EXAMPLE 2: Continuous Capture (Rolling Buffer)")
    print("=" * 70)
    
    processor = LiveCameraProcessor(camera_id=0)
    
    print("Continuous capture mode - simulating automatic direction detection")
    print("Will capture 50 frames and trigger prediction every 20 frames...")
    
    predictions_made = 0
    
    for i in range(50):
        # Capture frame
        frame = processor.capture_frame()
        if frame is None:
            print("✗ Failed to capture frame")
            break
        
        # Simulate direction detection at frame 10 and 30
        if i == 10:
            processor.set_intent(1)  # Left
        elif i == 30:
            processor.set_intent(2)  # Right
        
        # Make prediction every 20 frames (when buffer is full)
        if processor.is_ready() and (i + 1) % 20 == 0:
            video_tensor = processor.preprocess_current_buffer()
            predictions_made += 1
            
            intent_str = (CONFIG.class_names[processor.current_intent] 
                         if processor.current_intent is not None else "None")
            print(f"✓ Prediction {predictions_made} ready | Frame {i+1} | Intent: {intent_str}")
            
            # Clear intent after prediction
            processor.clear_intent()
        
        # Small delay to simulate real camera frame rate
        time.sleep(0.1)  # ~10 FPS
    
    processor.release()
    print(f"\n✓ Made {predictions_made} predictions from continuous capture")
    print()

def example_manual_frame_collection():
    """
    Example 3: Manual frame collection (without LiveCameraProcessor)
    
    Shows the low-level approach for maximum control
    """
    print("=" * 70)
    print("EXAMPLE 3: Manual Frame Collection")
    print("=" * 70)
    
    # Open camera directly
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("✗ Could not open camera")
        return
    
    # Collect frames manually
    frames = []
    print("Collecting 20 frames manually...")
    
    for i in range(CONFIG.seq_len):
        ret, frame = camera.read()
        if ret:
            frames.append(frame)
            print(f"  Frame {i+1}/{CONFIG.seq_len} collected", end='\r')
        else:
            print(f"\n✗ Failed to capture frame {i+1}")
            camera.release()
            return
    
    print(f"\n✓ All {CONFIG.seq_len} frames collected")
    
    camera.release()
    
    # Process frames
    preprocessor = VideoPreprocessor()
    video_tensor = preprocessor.preprocess_frame_sequence(
        frames=frames,
        intent=0,  # Front
        intent_position=5
    )
    
    print(f"✓ Preprocessed independently")
    print(f"  Output shape: {video_tensor.shape}")
    print()

def example_simulated_expo_app():
    """
    Example 4: Simulated Expo/React Native app workflow
    
    Demonstrates the typical flow in a mobile app:
    1. Start camera preview
    2. Buffer frames continuously
    3. Model detects upcoming turn direction
    4. Record exact frame when direction detected
    5. Collect remaining frames
    6. Provide audio feedback to user (Front/Left/Right)
    """
    print("=" * 70)
    print("EXAMPLE 4: Simulated Mobile App Workflow")
    print("=" * 70)
    
    processor = LiveCameraProcessor(camera_id=0)
    
    # Phase 1: Background frame capture (app camera preview)
    print("\n📱 Phase 1: Camera preview active, collecting frames...")
    for i in range(10):
        processor.capture_frame()
        time.sleep(0.1)
    print(f"   Buffer: {len(processor.frame_buffer)}/{CONFIG.seq_len} frames")
    
    # Phase 2: Model detects direction
    print("\n📱 Phase 2: Model detects LEFT turn direction")
    processor.set_intent(1)  # Left turn
    detection_frame = processor.frame_count
    
    # Phase 3: Continue capturing to fill buffer
    print("\n📱 Phase 3: Continuing capture to complete sequence...")
    while not processor.is_ready():
        processor.capture_frame()
        time.sleep(0.1)
    print(f"   Buffer: {len(processor.frame_buffer)}/{CONFIG.seq_len} frames")
    
    # Phase 4: Preprocess and predict
    print("\n📱 Phase 4: Preprocessing for prediction...")
    video_tensor = processor.preprocess_current_buffer()
    print(f"   Preprocessed shape: {video_tensor.shape}")
    
    # Phase 5: Send to model (simulated)
    print("\n📱 Phase 5: Sending to model for inference...")
    print("   [Model inference would happen here]")
    print("   Simulated prediction: LEFT (confidence: 0.89)")
    
    # Phase 6: Provide audio feedback to user
    print("\n📱 Phase 6: Providing audio feedback to user")
    print("   ✓ Audio instruction played: 'Turn LEFT'")
    
    processor.release()
    print("\n✓ Complete mobile app workflow demonstrated")
    print()

def example_performance_monitoring():
    """
    Example 5: Performance monitoring
    
    Measure preprocessing time and FPS for performance optimization
    """
    print("=" * 70)
    print("EXAMPLE 5: Performance Monitoring")
    print("=" * 70)
    
    processor = LiveCameraProcessor(camera_id=0)
    
    # Capture frames
    print("Capturing frames for performance test...")
    capture_start = time.time()
    
    for i in range(CONFIG.seq_len):
        processor.capture_frame()
    
    capture_time = time.time() - capture_start
    
    # Preprocess
    print("Preprocessing frames...")
    preprocess_start = time.time()
    
    processor.set_intent(1)
    video_tensor = processor.preprocess_current_buffer()
    
    preprocess_time = time.time() - preprocess_start
    
    # Calculate metrics
    total_time = capture_time + preprocess_time
    fps = CONFIG.seq_len / total_time
    
    print("\nPerformance Metrics:")
    print("-" * 70)
    print(f"  Capture time:     {capture_time*1000:.2f} ms")
    print(f"  Preprocess time:  {preprocess_time*1000:.2f} ms")
    print(f"  Total time:       {total_time*1000:.2f} ms")
    print(f"  Effective FPS:    {fps:.2f}")
    print(f"  Frames processed: {CONFIG.seq_len}")
    print("-" * 70)
    
    # Recommendations
    if total_time < 0.5:
        print("✓ Excellent performance. Suitable for real-time prediction")
    elif total_time < 1.0:
        print("✓ Good performance. Should work well on most devices")
    else:
        print("⚠ Performance may be slow on mobile devices")
        print("  Consider optimization or lower resolution")
    
    processor.release()
    print()

def example_error_handling():
    """
    Example 6: Error handling and edge cases
    
    Demonstrates proper error handling for production apps
    """
    print("=" * 70)
    print("EXAMPLE 6: Error Handling")
    print("=" * 70)
    
    try:
        # Try to open invalid camera
        print("Test 1: Invalid camera ID...")
        try:
            processor = LiveCameraProcessor(camera_id=99)
            print("  ✗ Should have raised error")
        except RuntimeError as e:
            print(f"  ✓ Caught expected error: {e}")
        
        # Try to preprocess with insufficient frames
        print("\nTest 2: Insufficient frames in buffer...")
        processor = LiveCameraProcessor(camera_id=0)
        try:
            processor.preprocess_current_buffer()
            print("  ✗ Should have raised error")
        except RuntimeError as e:
            print(f"  ✓ Caught expected error: {e}")
        
        # Proper cleanup
        print("\nTest 3: Proper resource cleanup...")
        processor.release()
        print("  ✓ Camera released successfully")
        
        print("\n✓ All error handling tests passed")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print()

def example_integration_template():
    """
    Example 7: Integration template for mobile apps
    
    Copy-paste template for integrating into Expo/React Native
    """
    print("=" * 70)
    print("EXAMPLE 7: Mobile App Integration Template")
    print("=" * 70)
    
    print("""
This is a template workflow for integrating into your Expo/React Native app:

// ============================================================================
// React Native / Expo Integration Pseudocode
// ============================================================================

// 1. INITIALIZATION (when app starts)
const cameraRef = useRef(null);
const frameBuffer = useRef([]);
const intentState = useState(null);

// 2. CAMERA PREVIEW (continuous frame capture)
const handleCameraFrame = (frame) => {
    // Add frame to buffer (keep last 20 frames)
    frameBuffer.current.push(frame);
    if (frameBuffer.current.length > 20) {
        frameBuffer.current.shift();  // Remove oldest
    }
};

// 3. DIRECTION DETECTION (model prediction)
const handleDirectionDetection = (direction) => {
    // Record detected direction and current frame index
    const intentPosition = frameBuffer.current.length - 1;
    intentState.set({
        direction: direction,      // 0=Front, 1=Left, 2=Right
        position: intentPosition
    });
    
    // Continue capturing until buffer is full (20 frames)
    // Then provide audio feedback to user
    waitForFullBuffer().then(provideAudioFeedback);
};

// 4. PREPROCESSING (call Python preprocessor or JavaScript equivalent)
const preprocess = async () => {
    // Option A: Call Python backend via API
    const response = await fetch('/api/preprocess', {
        method: 'POST',
        body: JSON.stringify({
            frames: frameBuffer.current,
            intent: intentState.direction,
            intent_position: intentState.position
        })
    });
    
    // Option B: Use JavaScript/TypeScript preprocessing (recommended)
    const preprocessor = new VideoPreprocessor();
    const tensor = preprocessor.preprocess_frame_sequence(
        frameBuffer.current,
        intentState.direction,
        intentState.position
    );
    
    return tensor;
};

// 5. AUDIO FEEDBACK (provide direction instruction)
const provideAudioFeedback = async () => {
    const tensor = await preprocess();
    const prediction = await runModel(tensor);  // ONNX or TFLite
    
    // Play audio instruction based on prediction
    const directions = ['Go straight', 'Turn left', 'Turn right'];
    playAudio(directions[prediction]);
};

// ============================================================================

Key Implementation Notes:
1. Use frame buffering to maintain rolling window of last 20 frames
2. Track EXACT frame index when model detects direction
3. Ensure frames are in BGR format (OpenCV standard) before preprocessing
4. For production, implement preprocessing in JavaScript/TypeScript
5. Use ONNX Runtime Mobile or TensorFlow Lite for on-device inference
6. Provide clear audio feedback for detected directions (Front/Left/Right)
7. Consider GPU acceleration for better performance

Files needed for mobile team:
- preprocessor.py (reference implementation)
- model_config.json (model parameters)
- best_convlstm.pth converted to ONNX or TFLite
- This example file (integration guide)
    """)
    
    print("=" * 70)
    print()

def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("LIVE CAMERA PROCESSING EXAMPLES")
    print("ConvLSTM Turn Prediction - Mobile App Module")
    print("=" * 70)
    print()
    
    print("⚠ IMPORTANT: These examples require a camera device")
    print("  If running in an environment without camera, some examples will fail")
    print()
    
    # Configuration info
    print("Configuration:")
    print(f"  Target FPS: {CONFIG.fps}")
    print(f"  Frames per sequence: {CONFIG.seq_len}")
    print(f"  Duration: {CONFIG.duration} seconds")
    print(f"  Classes: {', '.join(CONFIG.class_names)}")
    print()
    
    # Run examples
    try:
        example_basic_live_capture()
    except Exception as e:
        print(f"⚠ Example 1 skipped: {e}\n")
    
    try:
        example_continuous_capture()
    except Exception as e:
        print(f"⚠ Example 2 skipped: {e}\n")
    
    try:
        example_manual_frame_collection()
    except Exception as e:
        print(f"⚠ Example 3 skipped: {e}\n")
    
    try:
        example_simulated_expo_app()
    except Exception as e:
        print(f"⚠ Example 4 skipped: {e}\n")
    
    try:
        example_performance_monitoring()
    except Exception as e:
        print(f"⚠ Example 5 skipped: {e}\n")
    
    example_error_handling()
    
    example_integration_template()
    
    print("=" * 70)
    print("Examples Complete.")
    print("=" * 70)
    
if __name__ == "__main__":
    main()
