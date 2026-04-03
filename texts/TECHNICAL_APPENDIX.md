# EluSEEDate Technical Appendix (Source-Code Truth)

Prepared on: 2026-04-03
Purpose: implementation-grounded technical reference for maintenance and release hardening.

---

## 1. Scope

This appendix describes the code as it currently runs in the repository. It reflects active routes, runtime behavior, and data flow in:
- App.tsx
- src/navigation/types.ts
- src/screens/MainMenuScreen.tsx
- src/screens/ChoiceScreen.tsx
- src/screens/WayfindingScreen.tsx
- src/screens/ActiveCameraScreen.tsx
- src/screens/logsScreen.tsx
- src/services/preprocessor.ts
- src/services/convlstmWithoutIntentInference.ts
- src/services/convlstmWithIntentInference.ts
- src/services/yoloInference.ts
- src/services/geocodingService.ts
- src/services/directionsService.ts
- src/utils/imageUtils.ts
- src/config/modelConfig.ts

---

## 2. Runtime Architecture

The app currently has three main runtime branches:
1. Voice-first navigation and mode selection.
2. Destination capture and route construction (wayfinding).
3. Unified camera inference runtime (ActiveCamera) for both wandering and destination modes.

### 2.1 Root Navigation Contract

RootStackParamList:
1. MainMenu
2. Choice
3. Wayfinding
4. ActiveCamera
5. Logs

ActiveCamera route params:
1. mode: wandering | destination (required)
2. origin (optional)
3. destination (optional)
4. destinationLabel (optional)
5. routeSteps (optional)
6. totalDistanceMeters (optional)
7. totalDurationSeconds (optional)

### 2.2 User Flow

Flow A (Wandering):
1. MainMenu
2. Choice
3. ActiveCamera with mode=wandering

Flow B (Destination):
1. MainMenu
2. Choice
3. Wayfinding
4. ActiveCamera with mode=destination and route payload

Flow C (Diagnostics):
1. MainMenu
2. Logs

---

## 3. Voice and Wayfinding Pipeline

### 3.1 MainMenu and Choice

- MainMenu and Choice use react-native-vosk for command-style grammar.
- Choice supports Wandering, Destination, and Back commands.
- Choice now removes any previous Vosk result listener before attaching a new listener callback.

### 3.2 WayfindingScreen

Wayfinding is voice-first and uses:
- expo-location for user origin
- expo-speech for TTS prompts
- expo-speech-recognition for free-form destination and confirmation responses

High-level sequence:
1. Request location permission and resolve current location.
2. Prompt user to speak destination.
3. Geocode spoken destination.
4. Confirm candidate destination via yes/no/back.
5. Enforce MAX_RADIUS_KM = 10.
6. Fetch walking directions from OSRM service.
7. Navigate to ActiveCamera with destination payload.

Hardening behavior:
- Auto-restart voice-listening timeout is tracked and cleared during cleanup to avoid stale delayed restarts.

---

## 4. ActiveCamera Runtime

ActiveCamera is the single camera runtime for both modes.

### 4.1 Mode Selection

Pipeline is selected by:
- route.params.mode
- ENABLE_INTENT_MODE in modelConfig runtime settings

Selection matrix:
1. mode=wandering -> convlstmWithoutIntentInference
2. mode=destination and ENABLE_INTENT_MODE=false -> convlstmWithoutIntentInference
3. mode=destination and ENABLE_INTENT_MODE=true -> convlstmWithIntentInference

### 4.2 Camera Setup and Capture

- Camera implementation: expo-camera CameraView.
- Capture loop: recursive setTimeout (not setInterval).
- Capture frequency: REALISTIC_CAPTURE_FPS = 2.
- Capture method: takePictureAsync with quality 0.2, skipProcessing=true, shutterSound=false.
- Primary decode path: decodeImageUriToPixels.
- Fallback decode path: decodeBase64ToPixels.

Hardening changes in current source:
1. Camera-ready callback is guarded by refs so one-time picture-size configuration is not repeatedly re-run.
2. Capture gating checks ref-backed readiness flags to avoid stale-closure gating.
3. High-frequency console.log debug statements removed from capture and render paths.

### 4.3 Frame Buffer and ConvLSTM Input

Preprocessor output shape:
- [1, 20, 6, 128, 128]

Channel semantics:
1. Channels 0-2: RGB
2. Channels 3-5: intent channels

Current truth:
- Intent channels remain zero-filled unless explicit upstream intent injection is added.

Buffer behavior:
1. Early inference supported once minimum buffered frames are available.
2. First inference uses bootstrap doubling.
3. Subsequent inference uses tail padding.

---

## 5. YOLO Runtime

YOLO service uses react-native-fast-tflite when native runtime is available.

Current behavior:
1. Model loads from assets/model/yolo.tflite.
2. GPU delegate option is requested.
3. Output parsing supports transposed and non-transposed tensor layouts.
4. Detections are filtered by class allowlist and confidence threshold.
5. Same-class NMS is applied (IoU threshold 0.45).

Hardening updates applied:
1. Removed temporary high-volume debug logging from per-frame inference path.
2. Removed commented-out legacy tensor write blocks.
3. Enabled non-transposed tensor branch assignment in layout detection logic.

---

## 6. Configuration and Runtime Switches

In src/config/modelConfig.ts:
1. ConvLSTM preprocessing constants and model metadata are centralized.
2. Runtime switch MODEL_CONFIG.runtime.enableIntentMode controls destination-mode intent pipeline enablement.
3. ENABLE_INTENT_MODE is exported and consumed in ActiveCameraScreen.

---

## 7. Current Operational Notes

1. ActiveCamera is the only camera inference screen in active navigation flow.
2. Destination route payload is generated in Wayfinding and displayed in ActiveCamera performance overlay.
3. Logs screen is still available for diagnostics routing.
4. Release hardening removed temporary console.log noise in active capture and inference paths.

---

## 8. Recommended Ongoing Maintenance

1. Keep route and param changes mirrored in src/navigation/types.ts and this appendix.
2. Keep model input format comments synchronized with actual tensor write order in yoloInference.
3. Re-run TypeScript and Expo diagnostics before each release candidate.
4. Maintain changelog entries for each semantic version bump.

---

End of technical appendix.
