# EluSEEdate - Mobile Turn Prediction App

![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://shields.io/badge/JavaScript-F7DF1E?logo=JavaScript&logoColor=000&style=for-the-badge)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Expo](https://img.shields.io/badge/Expo-000020?style=for-the-badge&logo=expo&logoColor=white)

React Native/Expo mobile application for real-time turn direction prediction using a ConvLSTM deep learning model with TensorFlow Lite inference.

## Overview

This app uses two AI models working in parallel:

1. **ConvLSTM** (Convolutional Long Short-Term Memory) for turn direction prediction - analyzes sequences of video frames to predict if the user should go **Front**, **Left**, or **Right**
2. **YOLOv12** for real-time obstacle detection - identifies nearby objects (people, cars, bicycles, etc.) and displays bounding boxes on the camera view

Audio feedback is powered by **ObjectSpeechService**, which converts YOLO detections into spoken obstacle prompts with priority and cooldown controls.

**Turn Prediction Model**: Prototype 10 (ConvLSTM with Global Average Pooling)
**Obstacle Detection**: YOLOv12 with TFLite optimization
**Inference Engine**: TensorFlow Lite via `react-native-fast-tflite`

## Two Operating Modes

### 1. Demo Mode (Expo Go)
- **For quick UI/UX testing** without building native code
- Camera and all UI features work normally
- Predictions are **simulated** (random with realistic timing)
- Run with: `npx expo start`

### 2. Production Mode (Development Build)
- **Real TFLite inference** on device
- Actual model predictions from camera frames
- Requires native build
- Run with: `npx expo prebuild && npx expo run:android`

## Target Device

**Redmi Note 13 Pro 5G**
- Screen: 1080 x 2400 pixels (portrait)
- Camera: 200MP main (rear)
- Expected inference time: ~100ms

## Design

Minimalistic black & white palette for a clean, distraction-free interface.

## Features

- **Voice-first main menu** with Vosk commands (`Start`, `Exit`)
- **Mode selection flow** (`Wandering`, `Destination`, `Back`) with speech prompts
- **Wayfinding flow** for destination geocoding + spoken confirmation
- **Unified camera runtime** in `ActiveCameraScreen` for both wandering and destination pipelines
- **Live ConvLSTM turn prediction** with rolling frame buffer and low-latency updates
- **Live YOLO obstacle detection** with bounding box overlay
- **Priority-based spoken obstacle feedback** from YOLO detections (one object at a time, with cooldowns and danger interruption)
- **Performance overlay** with capture, preprocessing, ConvLSTM, and YOLO timings
- **Debug logs screen** for in-app runtime diagnostics

## Spoken Obstacle Feedback Rules

When multiple objects are detected in the same frame:
1. The app announces one object per cycle.
2. It prioritizes the largest bounding box (closest-on-screen proxy).
3. Ties are resolved by higher confidence, then higher danger weight.

Announcement pacing and priority:
1. Same-class cooldown: 5000 ms.
2. Global cooldown: 1500 ms.
3. Danger objects can interrupt lower-priority speech (for example car, bus, truck, train, motorcycle).
4. Speech call is asynchronous so camera/inference processing is not blocked.

## Route Graph

Current stack routes in `App.tsx`:

1. `MainMenu`
2. `Choice`
3. `Wayfinding`
4. `ActiveCamera`
5. `Logs`

Runtime flow:

1. `MainMenu` -> `Choice`
2. `Choice` -> `ActiveCamera` (wandering mode)
3. `Choice` -> `Wayfinding` -> `ActiveCamera` (destination mode with route payload)
4. `MainMenu` -> `Logs` (debug diagnostics)

## Project Structure

```
├── App.tsx                          # Main entry point
├── package.json                     # Dependencies
├── app.json                         # Expo configuration
├── tsconfig.json                    # TypeScript config
├── babel.config.js                  # Babel config
├── eas.json                         # EAS Build configuration
├── assets/
│   └── model/
│       ├── convlstm.tflite          # ConvLSTM TFLite model file
│       ├── convlstm.onnx            # ONNX model (backup)
│       └── yolo.tflite              # YOLOv12 TFLite model
├── texts/
│   ├── TECHNICAL_APPENDIX.md        # Source-code-truth architecture reference
│   ├── STANDALONE_APK_BUILD.txt     # EAS standalone APK guide
│   └── *.txt                        # Other reference docs
└── src/
    ├── components/
    │   ├── errorBoundary.tsx        # Error boundary component
    │   └── BoundingBoxOverlay.tsx   # YOLO bounding box renderer
    ├── config/
    │   └── modelConfig.ts           # ConvLSTM & YOLO configuration
    ├── navigation/
    │   └── types.ts                 # Navigation type definitions
    ├── screens/
  │   ├── MainMenuScreen.tsx       # Voice-first entry screen
  │   ├── ChoiceScreen.tsx         # Mode selection (Wandering/Destination)
  │   ├── WayfindingScreen.tsx     # Destination speech/geocoding flow
  │   ├── ActiveCameraScreen.tsx   # Unified camera + inference runtime
  │   └── LogsScreen.tsx           # Runtime log viewer
    ├── services/
    │   ├── preprocessor.ts          # Frame preprocessing (TypeScript)
    │   ├── convlstmWithoutIntentInference.ts  # ConvLSTM inference (no intent)
    │   ├── convlstmWithIntentInference.ts     # ConvLSTM inference (with intent, future)
  │   ├── yoloInference.ts         # YOLO object detection inference
  │   ├── geocodingService.ts      # Destination geocoding
  │   ├── directionsService.ts     # Walking route fetcher
  │   └── ObjectSpeechService.ts   # Spoken obstacle announcements
    └── utils/
        └── imageUtils.ts            # Image decoding utilities
```

## Model Configuration

### ConvLSTM Turn Prediction

| Parameter | Value |
|-----------|-------|
| Input Shape | [1, 20, 6, 128, 128] |
| Sequence Length | 20 frames |
| Model FPS | 20 frames/second (training) |
| Camera FPS | ~2 frames/second (actual capture rate) |
| Duration | 1 second of capture (20 frames @ 20 FPS) |
| Channels | 6 (3 RGB + 3 Intent) |
| Frame Size | 128 x 128 |
| Output Classes | 3 (Front, Left, Right) |
| Model Type | Float16 Quantized TFLite |
| Model Size | 3.62 MB (optimized from 7.14 MB) |
| GPU Acceleration | Enabled (via GPU delegate) |
| Expected Inference | ~100-200ms (GPU) / ~2-5s (CPU fallback) |

### YOLOv12 Obstacle Detection

| Parameter | Value |
|-----------|-------|
| Input Size | 128 x 128 (matches ConvLSTM, adjustable) |
| Model Type | TFLite (Float16 quantized expected) |
| Output | Bounding boxes with class probabilities |
| Classes | 80 (COCO dataset - adjust based on your model) |
| Confidence Threshold | 0.5 (50% minimum confidence) |
| GPU Acceleration | Enabled (via GPU delegate) |
| Expected Inference | ~30-100ms (faster than ConvLSTM) |
| Status | **Placeholder - awaiting real model** |

**Note**: Currently using a placeholder for YOLO. The app will simulate detections in demo mode until you add your actual YOLOv12 .tflite model to `assets/model/yolo.tflite`.

## Intent Channels

The model expects 6 channels per frame:
- Channels 0-2: RGB color channels [0, 1]
- Channels 3-5: Intent channels (all zeros for 'no intent' mode)

In this app, we always use "no intent" (all zeros for intent channels).

## Getting Started

### Prerequisites

- Node.js (v18 or v20 LTS recommended - v20.18.2 tested; **NOT v24+** due to compatibility issues with Expo SDK 50)
- Expo CLI
- Android Studio (for Android development)

### Installation

```bash
# Install dependencies
npm install

# Start Expo development server
npx expo start

# Run on Android device/emulator
npx expo run:android
```

### Building for Production

```bash
# Create production build using EAS Build (recommended)
npx eas build --platform android --profile preview

# For production release
npx eas build --platform android --profile production
```

## Usage

1. Launch the app — you will hear "Starting EluSEEdate" spoken aloud
2. Tap the **Start** button on the main menu (or say "Start")
3. Grant camera permission when prompted
4. Point the camera in the direction you're moving
5. The app will automatically:
   - Capture frames from the camera
   - Buffer frames until ready for prediction (min 10 frames)
   - Run predictions using the ConvLSTM model
   - Display the predicted direction at the bottom
   - Show performance metrics at the top-left

**Status Indicator**: A green dot means the app is actively capturing; "Demo Mode" label appears when using simulated predictions.

## Performance Metrics

The app displays the following metrics in the top-left corner:

- **Capture**: Time taken to capture a frame (in ms)
- **Inference**: Time taken by the TFLite model (in ms)
- **Preprocess**: Time taken to prepare frames (in ms)
- **Total**: Combined latency (in ms)
- **Frames/Predictions**: Count of captured frames and predictions

## Development Notes

### TFLite Implementation

The app uses `react-native-fast-tflite` for on-device TensorFlow Lite inference. This library:
- Provides GPU delegate support for accelerated inference
- Supports Float32 input tensors (required for ConvLSTM)
- Is registered as an Expo plugin in `app.json`

**Key Implementation Details** (see `src/services/inference.ts`):

```typescript
// Model loading (runs on app startup)
const model = await loadTensorflowModel(
  require('../../assets/model/convlstm.tflite')
);

// Inference (runs for each prediction)
const output = await model.run([tensorData]); // Float32Array [1, 20, 6, 128, 128]
// output[0] contains class probabilities [front, left, right]
```

**Demo Mode Fallback**: If TFLite isn't available (Expo Go), the app automatically switches to demo mode with simulated predictions. Check the status indicator on the camera screen.

### Frame Capture & Preprocessing

The CameraScreen captures frames using `expo-camera`:
- **Capture method**: `takePictureAsync` with base64 output (~200-500ms per frame)
- **Frame processing**: Decodes JPEG to pixel data, resizes to 128x128
- **Buffer management**: Rolling buffer of frames with padding for early predictions
- **Preprocessing**: Normalizes to [0,1], adds intent channels (zeros), transposes to NCHW format

**Preprocessing Pipeline** (see `src/services/preprocessor.ts`):
1. Camera captures JPEG frame → base64
2. Image decoded to RGBA pixel data
3. Resized to 128x128 using bilinear interpolation
4. Normalized to [0, 1] range
5. Intent channels added (all zeros)
6. Transposed to channels-first: [batch, seq, channels, height, width]

For production optimization, consider:
- Using `react-native-vision-camera` with frame processors for real-time 30 FPS capture
- Implementing native modules for direct YUV frame access
- Using GPU-accelerated preprocessing with `expo-gl` shaders

## Architecture

Based on **Prototype 10** - Mobile-Optimized ConvLSTM with Global Average Pooling + ONNX Export:
- 2-layer ConvLSTM: hidden_dim = [64, 32]
- Global Average Pooling (reduces model size by ~80%)
- Dropout (0.5) for regularization
- 3-class classification output

## License

Part of thesis project for ConvLSTM Turn Prediction.
