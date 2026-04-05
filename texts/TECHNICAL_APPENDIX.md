# EluSEEDate Technical Appendix (Source-Code Truth)

Prepared on: 2026-04-05
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
- src/screens/LogsScreen.tsx
- src/services/preprocessor.ts
- src/services/convlstmWithoutIntentInference.ts
- src/services/convlstmWithIntentInference.ts
- src/services/yoloInference.ts
- src/services/ObjectSpeechService.ts
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

Preprocessor channel allocation:
1. wandering path -> channels=3
2. destination path with ENABLE_INTENT_MODE=false -> channels=3
3. destination path with ENABLE_INTENT_MODE=true -> channels=6

### 4.2 Camera Setup and Capture

- Camera implementation: expo-camera CameraView.
- Capture loop: recursive setTimeout (not setInterval).
- Capture frequency: REALISTIC_CAPTURE_FPS = 2.
- Capture method: takePictureAsync with quality 0.2, skipProcessing=true, shutterSound=false.
- Primary decode path: decodeImageUriToPixels.
- Fallback decode path: decodeBase64ToPixels.

Destination-mode route progress wiring in current source:
1. ActiveCamera starts live GPS tracking using expo-location watchPositionAsync when ENABLE_INTENT_MODE is enabled.
2. ActiveCamera fetches and caches walking directions from current GPS position to route.params.destination.
3. Step advancement is distance-driven: when distance to current step end is less than 2 meters, currentStepIndex advances.
4. Distance checks use getGeoDistance imported from geolib/es/getDistance.
5. routeProgress keeps both distanceRemaining and distanceToStepEnd for intent-aware frame metadata.

Hardening changes in current source:
1. Camera-ready callback is guarded by refs so one-time picture-size configuration is not repeatedly re-run.
2. Capture gating checks ref-backed readiness flags to avoid stale-closure gating.
3. Structured runtime diagnostics are active in inference paths using INFERENCE-DEBUG, PRIORITY-DEBUG, and CONVLSTM-TRACE log families.
4. Performance overlay now includes audio diagnostics: current audio state and last announced object.
5. ActiveCamera cleanup now calls the selected ConvLSTM service cleanupModel on unmount.

### 4.3 Frame Buffer and ConvLSTM Input

Preprocessor output shape (dynamic):
1. [1, 20, 3, 128, 128] when lightweight path is active
2. [1, 20, 6, 128, 128] when intent-aware path is active

Channel semantics:
1. Channels 0-2: RGB
2. Channels 3-5: intent channels (only present in 6-channel path)

Current truth:
1. In 3-channel path, RGB is tightly packed and no intent slots are allocated.
2. In 6-channel path, RGB is written into positions 0-2 and positions 3-5 stay reserved for addIntent writes.
3. If ENABLE_INTENT_MODE is true, ActiveCamera writes per-frame intent metadata:
	- intent from maneuverToIntent(currentStep.maneuver)
	- intentDistance from routeProgress.distanceToStepEnd
4. Preprocessor addIntent writes intent channels per pixel:
	- if intentDistance <= 5 meters, set channel (3 + intentClass) to 1
	- otherwise, set Front intent channel (channel 3) to 1

Packing math used by preprocessor:
1. 3-channel path
	- frameStride = height * width * 3
	- pixel base index = frameOffset + (y * width + x) * 3
2. 6-channel path
	- frameStride = height * width * 6
	- pixel base index = frameOffset + (y * width + x) * 6

Buffer behavior:
1. Early inference supported once minimum buffered frames are available.
2. First inference uses bootstrap doubling.
3. Subsequent inference uses tail padding.

### 4.4 Unified Inference Diagnostics

Current ActiveCamera diagnostics:
1. YOLO emits INFERENCE-DEBUG count logs and PRIORITY-DEBUG closest-object logs using largest bounding-box area.
2. ConvLSTM emits INFERENCE-DEBUG and PRIORITY-DEBUG logs using highest class probability.
3. ConvLSTM emits CONVLSTM-TRACE logs for:
	- pipeline and buffer status at inference start
	- tensor readiness and preprocessing time
	- predicted label, confidence, and top probability ranking
	- timing summary (preprocess, inference, total latency, and FPS)
4. UI overlay surfaces the same runtime intent by showing:
	- Audio: Ready | Speaking | Error
	- Last Announced: most recent spoken object label

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

### 5.1 ObjectSpeech Runtime (Audio Obstacle Feedback)

ObjectSpeech is now wired directly into the ActiveCamera YOLO loop.

Integration flow:
1. ActiveCamera creates a single ObjectSpeechService instance in a stable useRef.
2. The service is started on screen mount.
3. Immediately after YOLO returns detections, ActiveCamera calls announceDetections(detections).
4. The call is non-blocking (fire-and-forget with Promise error catch) so camera/inference loop timing is not blocked by TTS.
5. The service is disposed on unmount to stop playback and avoid background audio leaks.
6. ActiveCamera subscribes to speech debug snapshots to drive overlay audio state and last-announced label.

Multi-object selection behavior:
1. Only one candidate is spoken per detection cycle.
2. Candidate filtering requires confidence > 0.5 and valid normalized box geometry.
3. Best candidate is selected by largest bounding-box area first (closest-on-screen proxy).
4. If area is tied, higher confidence wins.
5. If area and confidence are tied, higher danger weight wins.

Speech priority and anti-spam policy:
1. sameClassCooldownMs = 5000
2. globalCooldownMs = 1500
3. interruptPriorityDelta = 0.18
4. dangerInterruptThreshold = 0.9
5. High-danger objects (for example car, bus, truck, train, motorcycle) can pre-empt lower-priority active speech.

Audio hardening and tracing:
1. Service attempts local/offline voice selection using getAvailableVoicesAsync and language matching.
2. AUDIO-TRACE logs are emitted for service input and skip reasons (for example low confidence or cooldown).
3. Speech.speak is guarded with error capture, logging code and message when playback fails.
4. Audio runtime state is tracked as ready, speaking, or error.

Class name mapping for TTS:
1. Uses detection.className when meaningful.
2. If className is missing/placeholder (for example class_7), falls back to YOLO class lookup by classId.
3. Final spoken phrase is direction-aware: "<Object> ahead", "<Object> on the left", or "<Object> on the right".

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
4. Runtime diagnostics are intentionally verbose in inference and audio paths to support field debugging.
5. ConvLSTM currently has on-screen output (direction/confidence) and debug logs; spoken obstacle feedback is driven by YOLO detections through ObjectSpeechService.
6. ActiveCamera currently performs in-screen route distance checks using geolib/es/getDistance for step progression.

---

## 8. Recommended Ongoing Maintenance

1. Keep route and param changes mirrored in src/navigation/types.ts and this appendix.
2. Keep model input format comments synchronized with actual tensor write order in yoloInference.
3. Re-run npx expo-doctor, npx tsc --noEmit, and npx expo lint before each release candidate and before EAS preview/production builds.
4. Maintain changelog entries for each semantic version bump.

---

## 9. Troubleshooting Toolkit Results (2026-04-05)

Validation run performed on the current branch head after merge.

1. Expo Doctor
	- Command: npx expo-doctor
	- Result: 17/17 checks passed. No issues detected.

2. TypeScript No-Emit
	- Command: npx tsc --noEmit
	- Result: Completed with no type errors.

3. Expo Lint
	- Command: npx expo lint
	- Result: Completed with no lint errors or warnings.

Issue summary from this troubleshooting pass:
1. No current blocking or non-blocking issues were reported by the three toolkits.
2. No runtime source-code changes were required for Live GPS flow or preprocessor intent-channel behavior during this pass.

---

End of technical appendix.
