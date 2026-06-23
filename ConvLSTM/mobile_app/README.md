# 📱 Mobile App Preprocessor

**Version**: 1.0.1  
**Based On**: Prototype 9  
**Purpose**: Video preprocessing for ConvLSTM turn prediction on Android devices

---

## 🎯 Overview

This module provides preprocessing utilities for deploying the ConvLSTM turn prediction model on mobile Android devices using Expo/React Native. It replicates the exact preprocessing pipeline from Prototype 9 to ensure compatibility with the trained model.

**Supports Two Input Modes:**
- 📹 **Recorded Videos**: Process pre-recorded MP4 files for testing and offline analysis
- 📷 **Live Camera Feeds**: Process real-time camera frames for in-app predictions

### Key Features

✅ **Dual Input Modes**: Supports both recorded video files AND live camera feeds  
✅ **Frame Extraction**: Extracts frames at 10 FPS from video input  
✅ **Resize & Normalize**: Processes frames to 128×128 with [0,1] normalization  
✅ **Intent Encoding**: Adds 3 intent channels for direction signaling  
✅ **Mobile-Optimized**: Lightweight, efficient processing for resource-constrained devices  
✅ **Expo Compatible**: Designed for integration with React Native/Expo apps

---

## 📂 Module Structure

```
mobile_app/
├── __init__.py                  # Package initialization
├── config.py                    # Configuration constants (from Prototype 9)
├── preprocessor.py              # Main video preprocessing class
├── example_recorded_video.py    # Complete examples for pre-recorded videos
├── example_live_camera.py       # Complete examples for live camera feeds
├── CONCEPTUAL_NOTES.txt         # Key concepts explained (FPS, shapes, intent, etc.)
├── model_config.yaml            # Model configuration (YAML format)
├── model_config.json            # Model configuration (JSON format)
├── requirements.txt             # Python dependencies
├── FILES_FOR_MOBILE_TEAM.txt    # Deployment guide for mobile team
└── README.md                    # This file (API documentation)
```

---

## 🔧 Installation

### Prerequisites

Install all dependencies using the provided requirements file:

```bash
pip install -r mobile_app/requirements.txt
```

Or install individually:
```bash
pip install numpy>=1.19.0 opencv-python>=4.5.0 torch>=1.12.0
```

### Model Configuration

Two configuration files are provided for model parameters:
- **`model_config.yaml`**: Human-readable YAML format (recommended for Python)
- **`model_config.json`**: JSON format (recommended for JavaScript/TypeScript/Expo)

Both files contain identical configuration including:
- Model architecture parameters (input_dim, hidden_dim, etc.)
- Preprocessing settings (seq_len, fps, dimensions)
- Intent configuration
- Expected performance metrics
- Deployment settings

Load configuration in Python:
```python
import yaml
with open('mobile_app/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

Or in JavaScript:
```javascript
import config from './model_config.json';
const inputDim = config.model.input_dim;  // 6
```

### Usage in Python

```python
from mobile_app import VideoPreprocessor, preprocess_for_inference

# Option 1: Use convenience function
video_tensor = preprocess_for_inference("sample.mp4", intent=1)
# Output shape: [1, 20, 6, 128, 128] (ready for model)

# Option 2: Use class for more control
preprocessor = VideoPreprocessor()
video_tensor = preprocessor.preprocess_video("sample.mp4", intent=1, intent_position=5)
# Output shape: [20, 6, 128, 128]
```

---

## 📹 Input Modes: Recorded Videos vs Live Camera Feeds

### Mode 1: Recorded Video Files 🎥

**Best for**: Testing, offline processing, batch analysis

```python
from mobile_app import VideoPreprocessor

preprocessor = VideoPreprocessor()

# Process any video file (MP4, AVI, MOV, etc.)
video_tensor = preprocessor.preprocess_video(
    video_path="path/to/video.mp4",
    intent=1,  # 0=Front, 1=Left, 2=Right
    intent_position=5  # Optional: frame where intent starts
)

# Output: [20, 6, 128, 128] - Ready for inference
```

**How it works:**
- Opens video file with OpenCV
- Extracts 20 frames at 10 FPS (2 seconds total)
- Automatically handles different video frame rates
- Pads with zeros if video is too short

---

### Mode 2: Live Camera Feeds 📷

**Best for**: Real-time mobile app predictions, Expo/React Native integration

```python
from mobile_app import VideoPreprocessor
import cv2

preprocessor = VideoPreprocessor()

# Capture frames from camera (in the Expo app, this would be from device camera)
camera = cv2.VideoCapture(0)  # 0 = default camera
frames = []

for i in range(20):  # Collect 20 frames
    ret, frame = camera.read()
    if ret:
        frames.append(frame)  # BGR format from OpenCV

camera.release()

# Process the live frame sequence
video_tensor = preprocessor.preprocess_frame_sequence(
    frames=frames,  # List of 20 numpy arrays (BGR format)
    intent=2,  # Detected direction from model
    intent_position=None  # Auto-generate or specify
)

# Output: [20, 6, 128, 128] - Ready for inference
```

**How it works:**
- Takes pre-captured frames (One handle camera capture)
- Processes all 20 frames at once
- Perfect for real-time inference in mobile apps
- Must provide exactly 20 frames

---

### Which Mode Should You Use?

| Use Case | Mode | Method |
|----------|------|--------|
| Testing with sample videos | Recorded Video | `preprocess_video()` |
| Offline batch processing | Recorded Video | `preprocess_video()` |
| **Mobile app (Expo)** | **Live Camera** | **`preprocess_frame_sequence()`** |
| Real-time predictions | Live Camera | `preprocess_frame_sequence()` |
| Debugging/development | Recorded Video | `preprocess_video()` |

**For tue Expo Android app**: Use **`preprocess_frame_sequence()`** with live camera frames

---

## 📊 Preprocessing Pipeline

### Input
- **Video File**: MP4 or compatible format
- **Intent** (required for mobile): Direction intent (0=Front, 1=Left, 2=Right)
- **Intent Position** (required for mobile): Frame index when direction is detected
  - **IMPORTANT**: For mobile deployment, always specify the exact frame when the model detects the turn direction
  - Default fallback is frame 0, but production apps should track the exact timing

### Processing Steps

1. **Frame Extraction**
   - Extract frames at 10 FPS (configurable)
   - Total: 20 frames for 2-second sequence

2. **Resize**
   - Resize each frame to 128×128 pixels
   - Maintains aspect ratio with padding if needed

3. **Color Conversion**
   - Convert BGR (OpenCV format) to RGB
   - Model expects RGB input

4. **Normalization**
   - Normalize pixel values from [0, 255] to [0, 1]
   - Matches training preprocessing

5. **Channel Reordering**
   - Transpose from [H, W, C] to [C, H, W]
   - Channels-first format for PyTorch

6. **Intent Encoding**
   - Add 3 intent channels (one per direction)
   - If intent exists, fill corresponding channel with 1s for 1 second (10 frames)
   - Final channels: **3 RGB + 3 Intent = 6 total**

### Output
- **Shape**: `[seq_len, channels, height, width]` = `[20, 6, 128, 128]`
- **Type**: `numpy.ndarray` (float32)
- **Range**: [0, 1] for all channels

---

## 🎮 Intent Encoding

### Intent Types
| Value | Direction | Channel Index |
|-------|-----------|---------------|
| 0     | Front     | 3             |
| 1     | Left      | 4             |
| 2     | Right     | 5             |
| None  | No intent | All zeros     |

### Intent Detection

```
Frame Index:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
                       [Intent Position = 3]
                       └────────────┘
                    Intent Duration (10 frames = 1 second)

RGB Channels:   [Always present for all frames]
Intent Ch. 3:   0  0  0  1  1  1  1  1  1  1   1  1  1  1  0  0  0  0  0  0  (if intent=0)
Intent Ch. 4:   0  0  0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0  0  0  (zeros)
Intent Ch. 5:   0  0  0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0  0  0  (zeros)
```

### Intent Position Generation

**For Mobile Apps (Production):**
- Intent position should be explicitly provided based on when the model detects the turn direction
- Track the frame index when the direction is determined
- Pass this exact frame index to `intent_position` parameter

**Default Behavior (Testing Only):**
- If `intent_position` is not provided, defaults to frame 0 (start of video)
- This is a fallback for testing; production apps should always specify the position

**Note on Training vs Inference:**
- Prototype 9's random intent position generation was ONLY for training data augmentation
- For mobile inference, you want deterministic, user-controlled behavior
- The random generation has been removed from the mobile preprocessor

---

## 🔌 API Reference

### `VideoPreprocessor` Class

#### Constructor

```python
VideoPreprocessor(
    height: int = 128,
    width: int = 128,
    fps: int = 10,
    seq_len: int = 20,
    normalize: bool = True
)
```

**Parameters:**
- `height`: Target frame height (default: 128)
- `width`: Target frame width (default: 128)
- `fps`: Frames per second to extract (default: 10)
- `seq_len`: Number of frames in sequence (default: 20)
- `normalize`: Normalize pixels to [0, 1] (default: True)

#### Methods

##### `preprocess_video()`

```python
preprocess_video(
    video_path: Union[str, Path],
    intent: Optional[int] = None,
    intent_position: Optional[int] = None
) -> np.ndarray
```

Preprocess a video file for model inference.

**Parameters:**
- `video_path`: Path to video file
- `intent`: Direction intent (0: Front, 1: Left, 2: Right, None: no intent)
- `intent_position`: Frame index where intent starts (auto-generated if None)

**Returns:**
- `np.ndarray`: Shape `[20, 6, 128, 128]`

**Example:**
```python
preprocessor = VideoPreprocessor()
video_tensor = preprocessor.preprocess_video("sample.mp4", intent=1)
```

##### `preprocess_frame_sequence()`

```python
preprocess_frame_sequence(
    frames: list,
    intent: Optional[int] = None,
    intent_position: Optional[int] = None
) -> np.ndarray
```

Preprocess a sequence of already-extracted frames.

**Use Case**: Real-time camera feed processing

**Parameters:**
- `frames`: List of 20 numpy arrays (BGR format)
- `intent`: Direction intent
- `intent_position`: Frame index where intent starts

**Returns:**
- `np.ndarray`: Shape `[20, 6, 128, 128]`

**Example:**
```python
# Capture 20 frames from camera
frames = [...]  # List of 20 BGR frames

preprocessor = VideoPreprocessor()
video_tensor = preprocessor.preprocess_frame_sequence(frames, intent=2)
```

##### `get_output_shape()`

```python
get_output_shape() -> Tuple[int, int, int, int]
```

Get the expected output shape after preprocessing.

**Returns:**
- `Tuple`: `(20, 6, 128, 128)`

---

### `preprocess_for_inference()` Function

Convenience function for quick preprocessing with batch dimension added.

```python
preprocess_for_inference(
    video_path: Union[str, Path],
    intent: Optional[int] = None,
    intent_position: Optional[int] = None
) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Shape `[1, 20, 6, 128, 128]` (batch dimension added)

**Example:**
```python
from mobile_app import preprocess_for_inference

video_tensor = preprocess_for_inference("sample.mp4", intent=1)
# Ready to feed directly to model
prediction = model(video_tensor)
```

---

## 📱 Integration with Expo/React Native

### Architecture Overview

```
Android Device (Expo App)
    ↓
[Camera/Video Input]
    ↓
[JavaScript/TypeScript - React Native]
    ↓
[Bridge to Python Processing (optional)]
    ↓
[VideoPreprocessor - This Module]
    ↓
[Preprocessed Tensor: 20×6×128×128]
    ↓
[ONNX/TFLite Model Inference]
    ↓
[Prediction: Front/Left/Right]
    ↓
[Display Result in App]
```

### Integration

#### On-Device Processing (Production)
- Convert preprocessing to JavaScript/TypeScript
- Use TensorFlow.js or ONNX Runtime Mobile
- Process entirely on device
- No network required (better privacy & latency)

---

## 🔄 Model Compatibility

### Prototype 9 Model

This preprocessor is **100% compatible** with Prototype 9 trained models:

- **Input Shape**: `[Batch, 20, 6, 128, 128]`
- **Architecture**: 2-layer ConvLSTM with Global Average Pooling
- **Output**: 3 classes (Front, Left, Right)
- **Model Size**: ~1.5 MB (optimized for mobile)

### Loading Model

```python
import torch
import yaml
from models.conv_lstm_classifier import ConvLSTMModel

# Load configuration
with open('mobile_app/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model with config
model = ConvLSTMModel(
    input_dim=config['model']['input_dim'],             # 6
    hidden_dim=config['model']['hidden_dim'],           # [64, 32]
    kernel_size=tuple(config['model']['kernel_size']),  # (3, 3)
    num_layers=config['model']['num_layers'],           # 2
    height=config['model']['height'],                   # 128
    width=config['model']['width'],                     # 128
    num_classes=config['model']['num_classes']          # 3
)

# OR initialize manually (ensure values match model_config)
model = ConvLSTMModel(
    input_dim=6,
    hidden_dim=[64, 32],
    kernel_size=(3, 3),
    num_layers=2,
    height=128,
    width=128,
    num_classes=3
)

# Load trained weights
model.load_state_dict(torch.load("best_convlstm.pth"))
model.eval()

# Preprocess video
from mobile_app import preprocess_for_inference
video_tensor = preprocess_for_inference("test.mp4", intent=1)

# Convert to PyTorch tensor and predict
video_tensor = torch.from_numpy(video_tensor).float()
with torch.no_grad():
    prediction = model(video_tensor)
    predicted_class = prediction.argmax(dim=1).item()

print(f"Predicted direction: {['Front', 'Left', 'Right'][predicted_class]}")
```

---

## 🧪 Testing

### Unit Test Example

```python
from mobile_app import VideoPreprocessor, CONFIG

# Test initialization
preprocessor = VideoPreprocessor()
assert preprocessor.get_output_shape() == (20, 6, 128, 128)

# Test preprocessing (requires sample video)
video_tensor = preprocessor.preprocess_video("sample.mp4", intent=1, intent_position=5)
assert video_tensor.shape == (20, 6, 128, 128)
assert video_tensor.dtype == np.float32
assert video_tensor.min() >= 0.0 and video_tensor.max() <= 1.0

# Test intent channels
# Frame 5-14 should have intent signal in channel 4 (left)
for i in range(5, 15):
    assert video_tensor[i, 4, :, :].max() == 1.0  # Left intent channel active
    assert video_tensor[i, 3, :, :].max() == 0.0  # Front intent channel inactive
    assert video_tensor[i, 5, :, :].max() == 0.0  # Right intent channel inactive

print("All tests passed.")
```

---

## 📋 Requirements

### Python Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy>=1.19.0           # Array operations and preprocessing
opencv-python>=4.5.0    # Video frame extraction and processing
scipy>=1.6.0            # Intent position generation (normal distribution)
torch>=1.12.0           # Model loading and inference
torchvision>=0.13.0     # PyTorch utilities
onnx>=1.12.0            # Model conversion to ONNX format
onnxruntime>=1.12.0     # ONNX model inference
```

Install all dependencies:
```bash
pip install -r mobile_app/requirements.txt
```

### Mobile Device Requirements

- **OS**: Android 8.0+ (API Level 26+)
- **RAM**: Minimum 2 GB (4 GB recommended)
- **Storage**: ~10 MB for model + app
- **Camera**: For real-time processing (optional)

---

## 🚀 Expected Performance

| Device Type | Preprocessing Time | Inference Time | Total Latency |
|-------------|-------------------|----------------|---------------|
| High-end    | ~100ms            | ~50ms          | ~150ms        |
| Mid-range   | ~200ms            | ~100ms         | ~300ms        |
| Low-end     | ~400ms            | ~200ms         | ~600ms        |

*Times are estimates for 2-second video (20 frames)*

---

## 📚 Additional Resources

### Documentation
- [Prototype 9 README](../notebooks/PROTOTYPE_9_README.md) - Original model documentation
- [Mobile Deployment Guide](../SUBREADMES/MOBILE_DEPLOYMENT.md) - Full deployment instructions (if exists)
- [Model Architecture](../models/conv_lstm_classifier.py) - ConvLSTM implementation
- [Model Configuration](model_config.yaml) - Complete model and preprocessing config (YAML)
- [Model Configuration](model_config.json) - Complete model and preprocessing config (JSON)
- [Deployment Guide](FILES_FOR_MOBILE_TEAM.txt) - Complete list of files needed for mobile deployment
- [Conceptual Notes](CONCEPTUAL_NOTES.txt) - **START HERE** - Explains FPS, shapes, intent encoding, and other key concepts

### Example Files
- [example_recorded_video.py](example_recorded_video.py) - 7 complete examples for processing pre-recorded videos
- [example_live_camera.py](example_live_camera.py) - 7 complete examples for live camera feed processing with mobile integration template

---

## 📦 Files for Mobile Deployment

This module includes all files needed for mobile deployment:

| File | Description | Format |
|------|-------------|--------|
| `__init__.py` | Package initialization | Python |
| `config.py` | Configuration constants | Python |
| `preprocessor.py` | Video preprocessing class | Python |
| `example_recorded_video.py` | Examples for pre-recorded videos | Python |
| `example_live_camera.py` | Examples for live camera feeds | Python |
| `CONCEPTUAL_NOTES.txt` | Key concepts explained | Text |
| `model_config.yaml` | Model configuration | YAML |
| `model_config.json` | Model configuration | JSON |
| `requirements.txt` | Python dependencies | Text |
| `FILES_FOR_MOBILE_TEAM.txt` | Complete deployment guide | Text |
| `README.md` | API documentation (this file) | Markdown |

**For mobile team**: See [FILES_FOR_MOBILE_TEAM.txt](FILES_FOR_MOBILE_TEAM.txt) for complete deployment instructions and file list.

---

## 📝 License

Part of the ConvLSTM Turn Prediction thesis project.

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|| 1.0.1   | Feb 25, 2026 | Added example files (example_recorded_video.py, example_live_camera.py) || 1.0.0   | Feb 24, 2026 | Initial release based on Prototype 9 |

---