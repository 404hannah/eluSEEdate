# EluSEEDate Technical Appendix (Source-Code Truth)

Prepared on: 2026-04-03
Document role: Code-grounded technical reference for the current mobile app implementation.

---

## 1. Scope and Method

This appendix reflects implementation truth from the current codebase, not historical notes or planned behavior.

Scanned runtime files:
1. App.tsx
2. src/components/BoundingBoxOverlay.tsx
3. src/components/errorBoundary.tsx
4. src/config/index.ts
5. src/config/modelConfig.ts
6. src/navigation/index.ts
7. src/navigation/types.ts
8. src/screens/ChoiceScreen.tsx
9. src/screens/IntentScreen.tsx
10. src/screens/logsScreen.tsx
11. src/screens/MainMenuScreen.tsx
12. src/screens/NoIntentScreen.tsx
13. src/screens/WayfindingScreen.tsx
14. src/screens/index.ts
15. src/services/convlstmWithIntentInference.ts
16. src/services/convlstmWithoutIntentInference.ts
17. src/services/directionsService.ts
18. src/services/geocodingService.ts
19. src/services/index.ts
20. src/services/ObjectSpeechService.ts
21. src/services/preprocessor.ts
22. src/services/yoloInference.ts
23. src/utils/imageUtils.ts
24. src/utils/index.ts

Cross-audit references:
1. texts/VERSIONS.txt
2. texts/DATA_DICTIONARY.txt

---

## 2. Current Runtime Architecture

EluSEEDate currently contains three major runtime branches:
1. Voice-driven menu and mode selection.
2. Camera inference branch (ConvLSTM + YOLO) in Wandering and Destination modes.
3. Voice-driven wayfinding branch (GPS, geocoding, routing) before Destination mode.

High-level behavior:
1. MainMenu uses Vosk + TTS for Start/Exit commands.
2. Choice screen uses Vosk + TTS for Wandering/Destination/Back command routing.
3. Wayfinding uses expo-location and expo-speech-recognition (not Vosk) for free-form destination capture and yes/no/back confirmation.
4. NoIntentScreen and IntentScreen run camera capture plus ConvLSTM and YOLO pipelines.
5. LogsScreen globally intercepts console output and renders filterable in-app logs.

---

## 3. Navigation Contract and User Flow

Root stack routes (src/navigation/types.ts):
1. MainMenu
2. Choice
3. Wayfinding
4. Wandering
5. Destination (optional origin/destination/route payload)
6. Camera
7. Logs

Current route bindings in App.tsx:
1. Wandering -> NoIntentScreen
2. Destination -> IntentScreen
3. Camera -> NoIntentScreen (legacy alias)

Operational flows:

Flow A (Wandering):
1. MainMenu
2. Choice
3. Wandering

Flow B (Destination):
1. MainMenu
2. Choice
3. Wayfinding
4. Destination

Flow C (Debug):
1. MainMenu
2. Logs

---

## 4. Data and Inference Pipelines

## 4.1 Camera Capture Scheduling

Both NoIntentScreen and IntentScreen use:
1. takePictureAsync with low quality (0.1), base64 enabled, skipProcessing true.
2. REALISTIC_CAPTURE_FPS = 2.
3. Recursive setTimeout loop (not setInterval).

Loop timing model:
$$
\text{delay} = \max\left(0, \frac{1000}{\text{captureFps}} - \text{elapsedMs}\right)
$$

At 2 FPS, nominal capture interval is 500 ms.

## 4.2 FrameBuffer and Early Prediction

FrameBuffer behavior in preprocessor.ts:
1. samplingRate = max(1, floor(cameraFps / FPS)).
2. SEQ_LEN = 20.
3. canPredictEarly becomes true at ceil(20/2) = 10 frames.
4. getFramesPadded duplicates the last frame until length 20.
5. getFramesBootstrapDoubled duplicates each buffered frame in order (1,1,2,2,...) to bootstrap first prediction faster.

Both NoIntentScreen and IntentScreen use early prediction:
1. First prediction uses getFramesBootstrapDoubled.
2. Later predictions use getFramesPadded.

## 4.3 Tensorization and Channel Semantics

VideoPreprocessor output shape:
$$
[1, 20, 6, 128, 128]
$$

Channel semantics:
1. Channels 0-2: RGB from decoded RGBA input.
2. Channels 3-5: intent channels, currently left as zeros.

Index mapping used by preprocessor:
$$
\text{frameStride} = C \cdot H \cdot W
$$
$$
\text{frameOffset}(f) = f \cdot \text{frameStride}
$$
$$
\text{index}(f,c,y,x)=\text{frameOffset}(f)+c\cdot(H\cdot W)+y\cdot W+x
$$

Resize strategy in both image utility and preprocessor: nearest-neighbor.

## 4.4 ConvLSTM Inference Path (Current Truth)

NoIntentScreen and IntentScreen both import and call runPrediction from convlstmWithoutIntentInference.ts.

Current implication:
1. Destination mode does not currently use convlstmWithIntentInference.ts.
2. Intent channels are present in shape but remain zeros in practice.
3. No route-derived intent tensor is injected into ConvLSTM inputs yet.

ConvLSTM service behavior:
1. Attempts to load react-native-fast-tflite.
2. Falls back to demo mode if native module unavailable.
3. Runs model with one input tensor: model.run([tensor.data]).
4. Applies softmax and argmax over 3 classes.
5. Returns latency metrics:
   - preprocessingTimeMs
   - inferenceTimeMs
   - totalLatencyMs
   - fps

## 4.5 YOLO Inference Path

yoloInference.ts behavior:
1. Attempts to load yolo.tflite with GPU delegate.
2. Uses confidenceThreshold default 0.25 (service-level runtime setting).
3. Restricts classes using ALLOWED_CLASS_IDS.
4. Parses expected tensor cardinality based on 84 x 336 values.
5. Converts center-format boxes to top-left normalized boxes.
6. Applies same-class NMS with IoU threshold 0.45.

Important implementation note:
1. Preprocess comments describe BCHW, but current write logic uses contiguous RGB per pixel (BHWC-style flat ordering).
2. Source file contains unresolved merge conflict markers around a comment block (<<<<<<<, =======, >>>>>>>).

## 4.6 Object Speech Path

ObjectSpeechService behavior:
1. Picks one candidate per frame by area-first priority.
2. Uses confidence threshold and danger weights.
3. Applies same-class and global cooldowns.
4. Supports speech interruption only when priority gain exceeds threshold.

Current wiring truth:
1. IntentScreen instantiates ObjectSpeechService and stops it on back.
2. IntentScreen does not call announceDetections.
3. NoIntentScreen does not instantiate or call ObjectSpeechService.

Therefore, object speech announcements are not actively emitted in the current screen logic.

## 4.7 Wayfinding and Routing Pipeline

WayfindingScreen workflow:
1. Requests foreground location permission.
2. Gets current GPS coordinate.
3. Prompts user for destination via TTS.
4. Captures free-form spoken place name via expo-speech-recognition.
5. Resolves place with geocodeForward (Nominatim).
6. Reads back result and distance for yes/no/back confirmation.
7. Enforces MAX_RADIUS_KM = 10.
8. On confirmation, calls fetchWalkingDirections (OSRM).
9. Navigates to Destination with route payload.

Directions service output includes:
1. RouteStep[] with maneuver normalization.
2. one-hot maneuver vectors across 7 classes.
3. Distance/duration totals and route polyline.

Current gap:
1. Destination payload is passed to IntentScreen via navigation params.
2. IntentScreen currently does not consume these params.

---

## 5. File-by-File Technical Notes

## 5.1 App and Navigation

App.tsx:
1. Registers all stack screens.
2. Uses fade transitions and hidden headers.
3. Sets legacy Camera route to NoIntentScreen.

navigation/types.ts:
1. Includes expanded route contract for Destination params and Logs screen.

## 5.2 Screens

MainMenuScreen.tsx:
1. One-time startup greeting via module-level hasSpokenGreeting.
2. Vosk grammar: start, exit, [unk].

ChoiceScreen.tsx:
1. Voice and touch mode selection.
2. Vosk grammar: wandering, destination, back, [unk].

WayfindingScreen.tsx:
1. Voice-first destination capture and confirmation loop.
2. Uses expo-speech-recognition (not Vosk).
3. Handles geocoding failure and out-of-bounds recovery.

NoIntentScreen.tsx:
1. Camera + ConvLSTM + YOLO runtime path.
2. Early prediction enabled via FrameBuffer helper methods.

IntentScreen.tsx:
1. Operationally near-identical camera pipeline to NoIntentScreen.
2. Still imports no-intent ConvLSTM service.
3. Holds ObjectSpeechService ref but no announce call.

logsScreen.tsx:
1. Overrides global console methods at module load.
2. Stores last 500 logs.
3. Supports filtering by all/yolo/convlstm/errors.

## 5.3 Services

convlstmWithoutIntentInference.ts:
1. Active ConvLSTM inference service for both camera modes.

convlstmWithIntentInference.ts:
1. API-compatible parallel service exists.
2. Not currently wired into screen flow.

preprocessor.ts:
1. Adds FrameData.sequenceId for traceability.
2. Provides getFramesPadded and getFramesBootstrapDoubled.
3. Maintains persistent tensor buffer to reduce allocations.

yoloInference.ts:
1. Real parsing logic present.
2. Contains unresolved text-level merge markers.

directionsService.ts:
1. OSRM walking route fetch and maneuver canonicalization.

geocodingService.ts:
1. Nominatim forward and reverse geocoding.

ObjectSpeechService.ts:
1. Encapsulated priority-based TTS object alerting.

## 5.4 Components and Utilities

BoundingBoxOverlay.tsx:
1. Pure overlay projection of normalized detections.

errorBoundary.tsx:
1. Implemented but not mounted in App.tsx.

imageUtils.ts:
1. Base64 JPEG decode via jpeg-js.
2. Nearest-neighbor RGBA resizing.
3. Gray fallback image on decode failure.

---

## 6. Artifact Metrics (Current On-Disk)

Measured model artifact sizes in assets/model:
1. convlstm.tflite: 2,694,608 bytes (about 2.57 MiB)
2. yolo.tflite: 5,336,021 bytes (about 5.09 MiB)
3. convlstm.onnx: 569,600 bytes (about 0.54 MiB)

---

## 7. Documentation and Code Mismatch Audit (April 2026)

Mismatch 1:
1. preprocessor.ts header says 10 FPS sampling and shape [20,6,128,128].
2. Runtime flow uses 2 FPS capture in camera screens and tensor shape [1,20,6,128,128].

Mismatch 2:
1. DATA_DICTIONARY mentions bilinear resize.
2. Runtime uses nearest-neighbor resize in both imageUtils and preprocessor.

Mismatch 3:
1. DATA_DICTIONARY route list is outdated.
2. Actual stack now includes Choice, Wayfinding, Wandering, Destination, and Logs.

Mismatch 4:
1. DATA_DICTIONARY describes YOLO as placeholder/demo only.
2. yoloInference.ts contains active parsing logic and thresholding with runtime threshold default 0.25.

Mismatch 5:
1. VERSIONS and config metadata report different ConvLSTM size expectations.
2. On-disk convlstm.tflite is 2,694,608 bytes.

Mismatch 6:
1. Wayfinding payload (routeSteps and destination metadata) is produced.
2. Destination screen currently does not consume route params.

Mismatch 7:
1. Intent mode naming implies intent-aware ConvLSTM path.
2. IntentScreen currently uses convlstmWithoutIntentInference and zero intent channels.

Mismatch 8:
1. ObjectSpeechService exists and is instantiated in IntentScreen.
2. No announceDetections call is currently wired in camera screens.

Mismatch 9:
1. errorBoundary.tsx exists.
2. It is not mounted in App.tsx.

---

## 8. Code-State Risks and Portability Notes

Risk 1:
1. src/services/yoloInference.ts currently contains merge conflict markers in source text.
2. This should be resolved before release hardening.

Risk 2:
1. Several imports use filename casing that differs from disk casing.
2. This may pass on case-insensitive filesystems (Windows) but can fail on case-sensitive CI/build environments.

Risk 3:
1. LogsScreen globally patches console methods at module load.
2. This can affect performance and logging semantics outside the logs view.

---

## 9. Recommended Immediate Documentation and Engineering Updates

1. Update DATA_DICTIONARY.txt to match current route graph, resize method, YOLO runtime defaults, and screen/service names.
2. Update VERSIONS.txt with current measured artifact sizes.
3. Resolve merge markers in yoloInference.ts and normalize import filename casing.
4. Decide and document whether Destination mode should use convlstmWithIntentInference.ts and route-derived intent features.
5. If object speech is required in runtime, wire announceDetections in active camera screen logic and document thresholds/cooldowns.
6. Either mount ErrorBoundary in App.tsx or explicitly mark it as inactive tooling.

---

End of appendix update.