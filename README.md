# 🚗 ConvLSTM Turn Prediction Model

![Python](https://img.shields.io/badge/-Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)

A deep learning pipeline for assistive navigation that predicts maneuvers from video sequences using a ConvLSTM architecture. This model takes 3-second video clips and outputs turn predictions (Front, Left, Right) with user intent integration.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#1-training)
  - [Evaluation](#2-evaluation)
  - [TensorBoard Visualization](#3-visualizing-training-optional)
  - [Notebooks](#4-notebooks-optional)
  - [Mobile Deployment](#5-mobile-deployment-)
- [Model Architecture](#model-architecture)
- [Results](#results)

## 🎯 Project Overview

This repository implements a Convolutional LSTM network for real-time user maneuver prediction. The model:
- Processes 3-second video clips at 10 FPS (30 frames)
- Integrates optional intent signals
- Predicts three maneuver classes: Front (straight), Left turn, Right turn
- Operates at 128×128 resolution for efficient inference

### Key Features
- **Temporal Understanding**: LSTM cells capture motion patterns across time
- **Spatial Feature Extraction**: Convolutional layers detect road features
- **Intent Integration**: 3-channel driver intent signal
- **Mobile-Ready**: Complete preprocessing module for Android/Expo deployment with examples
- **Dynamic Cache Management**: LRU-based caching with auto-detection and warm caching
- **Memory Efficient Training**: Gradient accumulation and mixed precision training
- **Class Imbalance Handling**: Weighted loss function with automatic class weight calculation
- **Advanced Optimization**: Early stopping, learning rate scheduling, and gradient clipping

## 📂 Project Structure

```text
ConvLSTM/
├── models/
│   ├── __init__.py
│   └── conv_lstm_classifier.py     # Neural network architecture
│
├── mobile_app/                     # Mobile deployment module ⭐
│   ├── __init__.py
│   ├── config.py                   # Mobile configuration
│   ├── preprocessor.py             # Video preprocessing for mobile
│   ├── example_recorded_video.py   # Example: pre-recorded videos
│   ├── example_live_camera.py      # Example: live camera feeds
│   ├── model_config.yaml           # Model configuration (YAML)
│   ├── model_config.json           # Model configuration (JSON)
│   ├── requirements.txt            # Mobile dependencies
│   ├── FILES_FOR_MOBILE_TEAM.txt   # Deployment guide
│   └── README.md                   # Mobile API documentation
│
├── notebooks/                      # Development & experimentation
│   ├── prototype_1.ipynb           # Initial implementation
│   ├── prototype_2.ipynb           # Baseline version (3 channels)
│   ├── prototype_3-7.ipynb         # Progressive iterations
│   └── prototype_8.ipynb           # Current version ⭐
│
├── cache_manager.py                # Dynamic LRU cache management
├── dataset.py                      # PyTorch Dataset implementation
├── utils.py                        # Configuration & hyperparameters
├── train.py                        # Training script
├── tester.py                       # Evaluation script
├── best_convlstm.pth               # Trained model weights (generated)
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pandas numpy scikit-learn scipy
pip install tqdm pillow jupyter tensorboard
pip install split-folders  # For dataset splitting
```

## 📊 Dataset Preparation

### Expected Directory Structure
```text
Dataset_Folder/
├── videos/                         # 3-second video clips
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
│
└── labels/                         # Frame-level annotations
    ├── video_001_labels.csv        # Standard naming
    ├── video_002_labels_A.csv      # Alternative naming supported
    └── ...
```

### Video Specifications
- **Duration**: Exactly 3 seconds
- **Frame Rate**: 10 FPS (30 frames total)
- **Resolution**: 128×128 pixels
- **Format**: MP4

### Label CSV Format
Each CSV must contain a `label_id_corrected` column (or fallback to last column):
- `0`: Front/Straight
- `1`: Left turn
- `2`: Right turn

The model uses the last 24 frames (last second) for label determination.

## ⚙️ Configuration

### Update Paths (utils.py)
```python
VIDEO_DIR = r'C:\path\to\your\videos'
LABEL_DIR = r'C:\path\to\your\labels'
```

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `HEIGHT` | 128 | Frame height |
| `WIDTH` | 128 | Frame width |
| `CHANNELS` | 6 | Input channels (3 RGB + 3 intent) |
| `FPS` | 10 | Frames per second |
| `DURATION` | 3 | Clip duration (seconds) |
| `SEQ_LEN` | 30 | Total frames per sequence |

### Training Configuration (train.py)
```python
BATCH_SIZE = 2                      # Physical batch size
ACCUMULATION_STEPS = 4              # Gradient accumulation steps
                                    # Effective batch = 2 × 4 = 8
NUM_EPOCHS = 20                     # Training epochs
LEARNING_RATE = 1e-4                # Adam optimizer learning rate
SEED = 8                            # Random seed for reproducibility

# Regularization
DROPOUT_RATE = 0.5                  # Dropout for overfitting prevention
MAX_GRAD_NORM = 1.0                 # Gradient clipping threshold

# Early Stopping
EARLY_STOP_PATIENCE = 5             # Epochs without improvement before stopping
MIN_DELTA = 0.01                    # Minimum improvement threshold (%)

# Learning Rate Scheduler
LR_FACTOR = 0.5                     # LR reduction factor
LR_PATIENCE = 3                     # Epochs before LR reduction
LR_MIN = 1e-7                       # Minimum learning rate
```

### Model Parameters
```python
PARAMS = {
    'input_dim': 6,                 # 3 RGB + 3 intent channels
    'hidden_dim': [64, 32],         # LSTM hidden dimensions per layer
    'kernel_size': (3, 3),          # Convolutional kernel size
    'num_layers': 2,                # Number of LSTM layers
    'num_classes': 3,               # Front, Left, Right
    'dropout_rate': 0.5             # Dropout rate
}
```

### Cache Configuration
```python
CACHE_CONFIG = {
    'reserve_gb': 10.0,             # GB to keep free (auto-detect)
    'eviction_check_interval': 10,  # Check cache size every N misses
    'eviction_buffer_percent': 0.10 # Extra space to free (10%)
}
```

## 🚀 Usage

### 1. Training
Trains the model using 60/20/20 train/validation/test split:
```bash
python train.py
```

**Features:**
- **Automatic Dataset Splitting**: Uses `splitfolders` for 60/20/20 split
- **Dynamic Cache Management**: Auto-detects storage and pre-caches videos
- **Mixed Precision Training**: Reduces memory usage and speeds up training
- **Gradient Accumulation**: Effective batch size of 8 (2 × 4)
- **Class Balancing**: Automatic computation of class weights
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **TensorBoard Logging**: Real-time training visualization

**Output:**
- `best_convlstm.pth`: Best model based on validation accuracy
- `runs/`: TensorBoard logs for training visualization
- Console logs: Loss, accuracy, class weights per epoch
- Intent positions: `val_intent_positions.npy` and `test_intent_positions.npy`

### 2. Evaluation
Tests the trained model on the held-out test set:
```bash
python tester.py
```

**Output:**
- Performance metrics (accuracy, precision, recall, F1-score)
- Per-class classification report
- Latency measurements (avg ms per clip, FPS)
- `test_results.csv`: Detailed predictions

### 3. Visualizing Training (Optional)
View training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

### 4. Notebooks (Optional)
For experimentation and visualization:
```bash
jupyter notebook notebooks/prototype_8.ipynb
```

**Notebook Variants:**
- `prototype_8.ipynb`: **Current version** with all optimizations ⭐
- `prototype_1-7.ipynb`: Previous iterations and experiments

### 5. Mobile Deployment 📱
Deploy the model on Android devices using Expo/React Native:

**Quick Start:**
```bash
cd mobile_app
pip install -r requirements.txt
python example_recorded_video.py  # Test with recorded videos
python example_live_camera.py     # Test with live camera
```

**Documentation:**
- [`mobile_app/README.md`](mobile_app/README.md) - Complete API documentation
- [`mobile_app/FILES_FOR_MOBILE_TEAM.txt`](mobile_app/FILES_FOR_MOBILE_TEAM.txt) - Deployment guide
- [`example_recorded_video.py`](mobile_app/example_recorded_video.py) - 7 examples for pre-recorded videos
- [`example_live_camera.py`](mobile_app/example_live_camera.py) - 7 examples for live camera feeds

**Key Features:**
- ✅ Video file and live camera feed preprocessing
- ✅ Expo/React Native integration examples
- ✅ TensorFlow Lite conversion support
- ✅ Intent tracking and position encoding
- ✅ Performance monitoring and optimization tips

**Mobile Team Integration:**
See [`mobile_app/`](mobile_app/) folder for complete preprocessing pipeline, configuration files, and integration examples.

## 🧠 Model Architecture

### ConvLSTM Overview
The model combines spatial and temporal learning:

```
Input Video (30 frames)
    ↓
ConvLSTMCell Layer 1 (64 hidden dims)
    ↓
ConvLSTMCell Layer 2 (32 hidden dims)
    ↓
Final Hidden State (last frame)
    ↓
Fully Connected Layer
    ↓
Output (3 classes)
```

### Components

#### 1. ConvLSTMCell
- Processes one frame at a time
- Combines convolutional feature extraction with LSTM memory
- Maintains hidden state (h) and cell state (c)
- Uses 4 gates: input, forget, output, cell

#### 2. ConvLSTM Module
- Stacks multiple ConvLSTMCell layers
- Processes all 30 frames sequentially
- Returns final temporal representation

#### 3. ConvLSTMModel
- Wraps ConvLSTM + classification head
- Flattens final hidden state
- Linear layer maps to 3 class logits

### Key Architecture Components

#### Dynamic Cache Manager (New in Prototype 8)
Intelligent caching system for efficient video loading:
- **Auto-Detection**: Automatically detects available storage
- **Warm Caching**: Pre-caches videos before training starts
- **LRU Eviction**: Removes least recently used files when cache is full
- **O(1) Operations**: Constant-time cache lookups and updates

#### Training Optimizations
**Gradient Accumulation**:
```python
# Accumulate gradients over 4 steps (effective batch = 2 × 4 = 8)
for batch_idx, (data, targets) in enumerate(loader):
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Mixed Precision Training**:
```python
scaler = GradScaler()
with torch.autocast(device_type="cuda"):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

**Weighted Loss Function**:
```python
# Automatically calculated from training data
class_weights = compute_class_weights(train_labels)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### Intent Integration
The model accepts 6-channel input:
- **Channels 0-2**: RGB video frames
- **Channels 3-5**: Intent signals (Front, Left, Right)

Intent positions are determined based on user input patterns and saved as `.npy` files for reproducibility.

## 📈 Results

### Sample Output (from test_results.csv)
```csv
Actual_Label,Predicted_Label,Actual_Text,Predicted_Text,Correct
2,0,Right,Front,False
0,0,Front,Front,True
1,0,Left,Front,False
```

### Metrics Provided
- **Overall Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Latency**: Average inference time per clip (ms)
- **Throughput**: Clips processed per second

### Performance Notes
Based on the provided test results...

## 🔍 Troubleshooting

### Common Issues

**1. Directory Not Found**
```
WARNING: Directory not found: [PATH]
```
→ Update `VIDEO_DIR` and `LABEL_DIR` in `utils.py`

**2. CSV File Mismatch**
```
WARNING: Label file not found for video [NAME]
```
→ Ensure every `.mp4` has a matching `_labels.csv` or `_labels_A.csv`

**3. CUDA Out of Memory**
→ Reduce `BATCH_SIZE` in `train.py` or increase `ACCUMULATION_STEPS`
→ Clear cache with `torch.cuda.empty_cache()`

**4. Model File Not Found**
```
Model file not found at best_convlstm.pth
```
→ Run `train.py` first to generate the model

**5. Cache Size Issues**
```
⚠ Cache size exceeded
```
→ Adjust `reserve_gb` in cache configuration
→ Cache manager will automatically evict old files

**6. Split-folders Not Found**
```
ModuleNotFoundError: No module named 'split-folders'
```
→ Install with `pip install split-folders`

## 🔮 Future Work

- [ ] Implement data augmentation (rotation, brightness, etc.)
- [ ] Explore attention mechanisms for better temporal modeling
- [ ] Add online learning capabilities for continuous adaptation
- [ ] Optimize for edge deployment (quantization, pruning)
- [ ] Multi-modal fusion (LiDAR, GPS, IMU)
- [ ] Distributed training for larger datasets
- [ ] Real-time inference optimization

## 📝 Version History

### Prototype 8 (Current) ⭐
- Dynamic cache manager with LRU eviction and auto-detection
- Gradient accumulation for memory-efficient training
- Mixed precision training with torch.amp
- Weighted loss function with automatic class weight calculation
- Early stopping and learning rate scheduling
- Gradient clipping for stable training
- Centralized configuration management
- TensorBoard logging
- splitfolders integration for dataset splitting

### Prototype 7
- Enhanced training pipeline
- Improved data loading

### Prototype 1-6
- Progressive development and experimentation
- Intent integration implementation
- Model architecture refinement

---

**Note**: This is a research prototype. Real-world deployment requires extensive safety validation and testing.

