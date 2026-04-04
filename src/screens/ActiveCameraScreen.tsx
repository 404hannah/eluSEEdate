/*
 * Active Camera Screen - EluSEEdate
 *
 * Live camera view with real-time turn prediction
 * - Captures frames silently from rear camera (no sound/flash)
 * - Uses simplified capture approach compatible with Expo Go
 * - Supports two modes via route params:
 *   1) wandering: lightweight pipeline
 *   2) destination: intent-capable pipeline (if enabled globally)
 * - Shows predicted direction at bottom
 * - Shows inference/latency metrics at top-left
 *
 * NOTE: takePictureAsync is slow (~200-500ms), so we capture at 2-3 FPS
 * and duplicate frames to fill the 20-frame buffer for inference.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import {
  FrameBuffer,
  VideoPreprocessor,
  FrameData,
} from '../services/preprocessor';
import type {
  PredictionResult,
  PerformanceMetrics,
} from '../services/convlstmWithoutIntentInference';
import * as convlstmWithoutIntent from '../services/convlstmWithoutIntentInference';
import * as convlstmWithIntent from '../services/convlstmWithIntentInference';
import {
  runYOLODetection as detectObjects,
  initializeYOLOModel,
  YOLOResult,
  Detection,
} from '../services/yoloInference';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import {
  SEQ_LEN,
  FRAME_WIDTH,
  FRAME_HEIGHT,
  ENABLE_INTENT_MODE,
} from '../config/modelConfig';
import { decodeBase64ToPixels } from '../utils/imageUtils';
import { fetchWalkingDirections, maneuverToIntent, DirectionsResult } from '../services/directionsService';
import * as Location from 'expo-location';
import { getDistance as getGeoDistance } from 'geolib'; // npm install geolib

type ActiveCameraScreenProps = NativeStackScreenProps<RootStackParamList, 'ActiveCamera'>;

// Realistic capture FPS for takePictureAsync (slow but works in Expo Go)
const REALISTIC_CAPTURE_FPS = 2;

export default function ActiveCameraScreen({ navigation, route }: ActiveCameraScreenProps) {
  const mode = route.params.mode;
  const modeLabel = mode === 'destination' ? 'Destination' : 'Wandering';
  const useIntentPipeline = mode === 'destination' && ENABLE_INTENT_MODE;
  const convlstmService = useIntentPipeline ? convlstmWithIntent : convlstmWithoutIntent;

  const destinationLabel = route.params.destinationLabel;
  const routeStepCount = route.params.routeSteps?.length ?? 0;
  const totalDistanceMeters = route.params.totalDistanceMeters;

  // Wayfinding state (live GPS + directions)
  const [userLocation, setUserLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [directionsCache, setDirectionsCache] = useState<DirectionsResult | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);

  // Camera permission state
  const [permission, requestPermission] = useCameraPermissions();
  
  // Camera reference for frame capture
  const cameraRef = useRef<CameraView>(null);
  
  // Camera mount state
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  
  // Frame buffer for storing captured frames (use realistic FPS)
  const frameBufferRef = useRef<FrameBuffer>(new FrameBuffer(REALISTIC_CAPTURE_FPS));
  
  // Preprocessor instance
  const preprocessorRef = useRef<VideoPreprocessor>(new VideoPreprocessor());
  
  // Prediction state
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [directionLabel, setDirectionLabel] = useState<string>('Waiting...');
  const [confidence, setConfidence] = useState<number>(0);
  
  // Performance metrics state
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    preprocessingTimeMs: 0,
    inferenceTimeMs: 0,
    totalLatencyMs: 0,
    fps: 0,
  });
  
  // Processing state
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [frameCount, setFrameCount] = useState<number>(0);
  const [predictionCount, setPredictionCount] = useState<number>(0);
  const [debugStatus, setDebugStatus] = useState<string>(`Initializing ${modeLabel} mode...`);
  const [lastCaptureTime, setLastCaptureTime] = useState<number>(0);
  
  // YOLO detection state
  const [isYOLOModelLoaded, setIsYOLOModelLoaded] = useState<boolean>(false);
  const [yoloDetections, setYoloDetections] = useState<Detection[]>([]);
  const [yoloInferenceTime, setYoloInferenceTime] = useState<number>(0);
  
  // Inference lock to prevent concurrent inferences
  const isInferencingRef = useRef<boolean>(false);
  const isCapturingRef = useRef<boolean>(false);
  const isYOLOInferencingRef = useRef<boolean>(false);
  const isModelLoadedRef = useRef<boolean>(false);
  const isYOLOModelLoadedRef = useRef<boolean>(false);
  const hasFirstPredictionRef = useRef<boolean>(false);
  const captureSequenceRef = useRef<number>(0);
  const predictionCountRef = useRef<number>(0);
  
  // Capture interval reference (now using setTimeout for async control)
  const captureIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Initialize model on screen mount
   */
  useEffect(() => {
    console.log('[ActiveCamera] Screen mounted (mode=' + mode + ')');
    console.log('[ActiveCamera] Permission status:', permission?.granted ? 'granted' : 'not granted');
    
    const initModels = async () => {
      console.log('[ActiveCamera] Initializing models for', modeLabel, 'mode...');
      setDebugStatus('Loading models...');
      
      // Initialize ConvLSTM model
      const convlstmLoaded = await convlstmService.initializeModel();
      setIsModelLoaded(convlstmLoaded);
      isModelLoadedRef.current = convlstmLoaded;
      if (convlstmLoaded) {
        console.log(
          '[ActiveCamera] ConvLSTM initialized successfully (' +
            (useIntentPipeline ? 'intent pipeline' : 'wandering pipeline') +
            ')'
        );
      } else {
        console.log('[ActiveCamera] ConvLSTM model failed to load');
      }
      
      // Initialize YOLO model
      const yoloLoaded = await initializeYOLOModel();
      setIsYOLOModelLoaded(yoloLoaded);
      isYOLOModelLoadedRef.current = yoloLoaded;
      if (yoloLoaded) {
        console.log('[ActiveCamera] YOLO model initialized successfully');
      } else {
        console.log('[ActiveCamera] YOLO model failed to load');
      }

      if (convlstmLoaded && yoloLoaded) {
        setDebugStatus('Models ready');
      } else if (!convlstmLoaded && !yoloLoaded) {
        setDebugStatus('Model load failed: ConvLSTM + YOLO unavailable');
      } else if (!convlstmLoaded) {
        setDebugStatus('Model load failed: ConvLSTM unavailable');
      } else {
        setDebugStatus('Model load failed: YOLO unavailable');
      }
    };
    
    initModels();
    
    let locationSub: Location.LocationSubscription | null = null;

    if (ENABLE_INTENT_MODE) {
      (async () => {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status === 'granted') {
          locationSub = await Location.watchPositionAsync(
            { accuracy: Location.Accuracy.High, distanceInterval: 2 },
            (loc) => {
              setUserLocation({
                latitude: loc.coords.latitude,
                longitude: loc.coords.longitude,
              });
            },
          );
        } else {
          console.warn('Location permission denied');
        }
      })();
    }
    
    // Cleanup on unmount — use refs only, no state updates
    return () => {
      isCapturingRef.current = false;
      if (captureIntervalRef.current) {
        clearTimeout(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
      if (locationSub) {
        locationSub.remove();
      }
    };
  }, []);

  /**
   * Start continuous frame capture when permission is granted
   * Start regardless of model status (demo mode works too)
   */
  useEffect(() => {
    if (permission?.granted) {
      // Small delay to ensure camera is ready
      const timer = setTimeout(() => {
        startCapture();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [permission?.granted]);

  useEffect(() => {
    isModelLoadedRef.current = isModelLoaded;
  }, [isModelLoaded]);

  useEffect(() => {
    isYOLOModelLoadedRef.current = isYOLOModelLoaded;
  }, [isYOLOModelLoaded]);

  // Directions cache
  useEffect(() => {
    if (mode === 'destination' && userLocation && route.params.destination) {
      const destination = route.params.destination; // { latitude, longitude }
      const fetchAndCacheDirections = async () => {
        try {
          const directions = await fetchWalkingDirections(userLocation, destination);
          setDirectionsCache(directions);
          setCurrentStepIndex(0); // Start at first step
          console.log('Directions cached:', directions.steps.length, 'steps');
        } catch (error) {
          console.warn('Failed to fetch directions:', error);
        }
      };
      fetchAndCacheDirections();
    }
  }, [mode, userLocation, route.params.destination]);

  // Location Watcher
  useEffect(() => {
    if (!ENABLE_INTENT_MODE) return;
    
    let subscription: Location.LocationSubscription | null = null;
    
    const startLocationWatching = async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        console.warn('Location permission denied');
        return;
      }
      
      // Watch position with reasonable accuracy and update frequency
      subscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          timeInterval: 2000,        // Update every 2 seconds
          distanceInterval: 5,        // Or every 5 meters walked
        },
        (location) => {
          setUserLocation({
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
          });
          console.log('[Location] Updated:', location.coords);
        }
      );
    };
    
    startLocationWatching();
    
    return () => {
      subscription?.remove();
    };
  }, []);

  useEffect(() => {
    if (!directionsCache || !userLocation || !ENABLE_INTENT_MODE) return;
    
    const steps = directionsCache.steps;
    if (currentStepIndex >= steps.length) return;
    
    const checkAndAdvanceStep = () => {
      const currentStep = steps[currentStepIndex];
      const distanceToEnd = getGeoDistance(userLocation, currentStep.endLocation);
      
      if (distanceToEnd < 5) { // Within 5 meters
        setCurrentStepIndex(prev => Math.min(prev + 1, steps.length - 1));
        console.log(`[Route] Advanced to step ${currentStepIndex + 1}`);
      }
    };
    
    checkAndAdvanceStep();
  }, [userLocation, directionsCache, currentStepIndex]);

  const [routeProgress, setRouteProgress] = useState({
    currentStepIndex: 0,
    distanceRemaining: 0,
    distanceToStepEnd: 0,
    stepsCompleted: 0,
  });

  useEffect(() => {
    if (!directionsCache || !userLocation) return;
    
    // Calculate remaining distance
    let remainingDist = 0;
    for (let i = currentStepIndex; i < directionsCache.steps.length; i++) {
      remainingDist += directionsCache.steps[i].distanceMeters;
    }
    
    // Distance to current step end
    const currentStep = directionsCache.steps[currentStepIndex];
    const distToEnd = getGeoDistance(userLocation, currentStep.endLocation);
    
    setRouteProgress({
      currentStepIndex,
      distanceRemaining: remainingDist,
      distanceToStepEnd: distToEnd,
      stepsCompleted: currentStepIndex,
    });
  }, [userLocation, directionsCache, currentStepIndex]);

  /**
   * Start continuous frame capture
   */
  const startCapture = useCallback(() => {
    if (captureIntervalRef.current || isCapturingRef.current) return;
    
    isCapturingRef.current = true;
    hasFirstPredictionRef.current = false;
    predictionCountRef.current = 0;
    setPredictionCount(0);
    setIsCapturing(true);
    setDebugStatus('Starting capture...');
    console.log(`[ActiveCamera] Starting continuous capture at ${REALISTIC_CAPTURE_FPS} FPS...`);

    // Use recursive timeout instead of setInterval for proper async handling
    const captureLoop = async () => {
      if (!isCapturingRef.current) return;
      
      const startTime = Date.now();
      await captureFrame();
      const elapsed = Date.now() - startTime;
      
      // Schedule next capture, accounting for time spent
      const captureInterval = 1000 / REALISTIC_CAPTURE_FPS;
      const delay = Math.max(0, captureInterval - elapsed);
      
      captureIntervalRef.current = setTimeout(captureLoop, delay) as any;
    };
    
    // Start the loop
    captureLoop();
  }, []);

  /**
   * Stop frame capture
   */
  const stopCapture = useCallback(() => {
    if (!isCapturingRef.current && !captureIntervalRef.current) return; // already stopped
    isCapturingRef.current = false;
    hasFirstPredictionRef.current = false;
    if (captureIntervalRef.current) {
      clearTimeout(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setIsCapturing(false);
    setDebugStatus('Capture stopped');
    console.log('[ActiveCamera] Capture stopped');
  }, []);

  /**
   * Capture a single frame from camera
   * Uses takePictureAsync which is slow but works in Expo Go
   */
  const captureFrame = async () => {
    if (!cameraRef.current) {
      return;
    }
    
    const startTime = Date.now();
    
    try {
      setDebugStatus('Capturing frame...');
      
      // Capture frame silently (no sound, no animation)
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.1,           // Low quality for faster capture
        base64: true,           // Get base64 for processing
        skipProcessing: true,
        shutterSound: false,    // Disable shutter sound
      });
      
      if (!photo) {
        console.log('[ActiveCamera] No photo returned');
        setDebugStatus('Capture failed - no photo');
        return;
      }
      
      const captureTime = Date.now() - startTime;
      setLastCaptureTime(captureTime);
      console.log(`[ActiveCamera] Frame captured in ${captureTime}ms`);
      
      // Decode base64 image to pixel data
      let frameData: FrameData;
      
      if (photo.base64) {
        try {
          // Decode the base64 image to actual pixel data
          const decoded = await decodeBase64ToPixels(photo.base64, FRAME_WIDTH, FRAME_HEIGHT);
          const sequenceId = ++captureSequenceRef.current;
          
          let intent = -1;
          let intentDistance = 0;
        
          if (ENABLE_INTENT_MODE && directionsCache && userLocation && routeProgress) {
            // Set intent for current step
            const currentStep = directionsCache.steps[currentStepIndex];
            if (currentStep) {
              intent = maneuverToIntent(currentStep.maneuver);
              intentDistance = routeProgress.distanceToStepEnd;
            }
          }

          // Can I add intent here if YOLO accepts this as data
          frameData = {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            timestamp: Date.now(),
            sequenceId,
            intent,
            intentDistance
          };
          
          console.log(`[ActiveCamera] Frame decoded #${sequenceId}: ${decoded.width}x${decoded.height}, ${decoded.data.length} bytes`);
        } catch (decodeError: any) {
          console.warn('[ActiveCamera] Failed to decode image:', decodeError?.message);
          setDebugStatus(`Decode error: ${decodeError?.message || 'invalid frame'}`);
          return;
        }
      } else {
        // No base64 data available.
        console.warn('[ActiveCamera] No base64 data in photo');
        setDebugStatus('Capture error: no base64 frame data');
        return;
      }
      
      // Add frame to buffer
      const wasAdded = frameBufferRef.current.addFrame(frameData);
      
      if (wasAdded) {
        // Use functional update to avoid stale closure
        setFrameCount(prev => {
          const newCount = prev + 1;
          const buffer = frameBufferRef.current;
          const bufferCount = buffer.getFrameCount();
          setDebugStatus(`Captured: ${newCount} | Buffer: ${bufferCount}/${SEQ_LEN}`);
          console.log(`[ActiveCamera] Frame ${newCount} added to buffer (${bufferCount}/${SEQ_LEN}) seq=${frameData.sequenceId ?? -1}`);
          return newCount;
        });
        
        // Run inference when buffer is ready (or can predict early with padding)
        const buffer = frameBufferRef.current;
        if (isModelLoadedRef.current && buffer.canPredictEarly() && !isInferencingRef.current) {
          await runInferenceWithPadding();
        }
        
        // Run YOLO detection on this frame (parallel with ConvLSTM)
        if (isYOLOModelLoadedRef.current && !isYOLOInferencingRef.current) {
          await runYOLODetection(frameData);
        }
      }
    } catch (error: any) {
      console.error('[ActiveCamera] Frame capture error:', error?.message || error);
      setDebugStatus(`Error: ${error?.message || 'capture failed'}`);
    }
  };
  
  /**
   * Run YOLO object detection on single frame
   */
  const runYOLODetection = async (frame: FrameData) => {
    if (!isYOLOModelLoadedRef.current) {
      return;
    }

    if (isYOLOInferencingRef.current) {
      return; // Skip if already running
    }
    
    isYOLOInferencingRef.current = true;
    
    try {
      const result: YOLOResult = await detectObjects(frame);
      setYoloDetections(result.detections);
      setYoloInferenceTime(result.inferenceTimeMs);
    } catch (error: any) {
      console.error('[YOLO] Detection error:', error?.message || error);
      setYoloDetections([]);
      setDebugStatus(`YOLO error: ${error?.message || 'inference failed'}`);
    } finally {
      isYOLOInferencingRef.current = false;
    }
  };

  /**
   * Run model inference with frame padding if buffer not full
   */
  const runInferenceWithPadding = async () => {
    if (!isModelLoadedRef.current) {
      return;
    }

    const buffer = frameBufferRef.current;
    
    if (isInferencingRef.current) {
      return;
    }
    
    isInferencingRef.current = true;
    setDebugStatus('Running inference...');
    
    try {
      // Before first prediction, duplicate each buffered frame in order
      // (1,1,2,2,...) to bootstrap sequence length faster.
      const isFirstPrediction = !hasFirstPredictionRef.current;
      const strategy = isFirstPrediction ? 'bootstrap-doubled' : 'tail-padded';
      const frames = isFirstPrediction
        ? buffer.getFramesBootstrapDoubled()
        : buffer.getFramesPadded();

      if (frames.length !== SEQ_LEN) {
        throw new Error(`Invalid frame sequence length: ${frames.length}`);
      }

      const sequenceIds = frames.map((f, idx) => f.sequenceId ?? -(idx + 1));
      const uniqueIds = Array.from(new Set(sequenceIds));
      const runTag = predictionCountRef.current + 1;
      console.log(
        `[TRACK][ActiveCamera][${mode}][Run ${runTag}] strategy=${strategy} buffer=${buffer.getFrameCount()}/${SEQ_LEN} unique=${uniqueIds.length}/${frames.length}`
      );
      console.log(`[TRACK][ActiveCamera][${mode}][Run ${runTag}] sequence=${sequenceIds.join(',')}`);
      
      // Preprocess frames
      const preprocessor = preprocessorRef.current;
      const tensor = preprocessor.preprocessFrameSequence(frames);
      
      // Run prediction
      const { prediction, metrics: newMetrics } = await convlstmService.runPrediction(tensor);
      
      // Update state
      setCurrentPrediction(prediction);
      setDirectionLabel(prediction.className);
      setConfidence(prediction.confidence);
      setMetrics(newMetrics);
      hasFirstPredictionRef.current = true;
      
      // Use functional update to avoid stale closure
      setPredictionCount(prev => {
        const newPredCount = prev + 1;
        predictionCountRef.current = newPredCount;
        setDebugStatus(`Prediction #${newPredCount}: ${prediction.className}`);
        console.log(`[ActiveCamera] Prediction #${newPredCount}: ${prediction.className} (${(prediction.confidence * 100).toFixed(1)}%)`);
        console.log(`[ActiveCamera] Latency: ${newMetrics.totalLatencyMs.toFixed(1)}ms`);
        return newPredCount;
      });
      
    } catch (error: any) {
      console.error('[ActiveCamera] Inference error:', error?.message || error);
      setDebugStatus(`Inference error: ${error?.message || 'unknown'}`);
    } finally {
      isInferencingRef.current = false;
    }
  };

  /**
   * Handle back button press
   */
  const handleBack = () => {
    stopCapture();
    if (navigation.canGoBack && navigation.canGoBack()) {
      navigation.goBack();
    } else if (navigation.navigate) {
      navigation.navigate('MainMenu');
    }
  };

  // Permission not determined yet
  if (!permission) {
    console.log('[ActiveCamera] Permission not determined yet');
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </SafeAreaView>
    );
  }

  // Permission denied
  if (!permission.granted) {
    console.log('[ActiveCamera] Permission denied');
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera access is required</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }
  
  console.log('[ActiveCamera] Rendering camera view - permission granted');

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      
      {/* Camera View - Silent capture mode (no children allowed) */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        mode="picture"
        animateShutter={false}
        enableTorch={false}
        onCameraReady={() => {
          console.log('[ActiveCamera] Camera is ready!');
          setIsCameraReady(true);
          setDebugStatus('Camera ready');
        }}
        onMountError={(error) => {
          console.error('[ActiveCamera] Mount error:', error);
          setDebugStatus(`Camera error: ${error.message}`);
        }}
      />
      
      {/* YOLO Bounding Boxes Overlay */}
      <BoundingBoxOverlay 
        detections={yoloDetections}
        containerWidth={Dimensions.get('window').width}
        containerHeight={Dimensions.get('window').height}
      />
      
      {/* Overlay Container - Absolute positioned on top of camera */}
      <View style={styles.overlayContainer}>
        {/* Debug Camera Status - Center of screen */}
        {!isCameraReady && (
          <View style={styles.cameraStatusOverlay}>
            <Text style={styles.cameraStatusText}>📷 Initializing Camera...</Text>
            <Text style={styles.cameraStatusSubtext}>Please wait</Text>
          </View>
        )}
        
        {/* Performance Overlay (Top-Left) */}
        <View style={styles.performanceOverlay}>
          <Text style={styles.performanceTitle}>Performance</Text>
          <Text style={styles.performanceText}>
            Mode: {modeLabel}
          </Text>
          <Text style={styles.performanceText}>
            ConvLSTM: {useIntentPipeline ? 'Intent pipeline' : 'Wandering pipeline'}
          </Text>
          {mode === 'destination' && destinationLabel ? (
            <Text style={styles.performanceText} numberOfLines={2}>
              Target: {destinationLabel}
            </Text>
          ) : null}
          {mode === 'destination' ? (
            <Text style={styles.performanceText}>
              Steps: {routeStepCount} | Dist: {totalDistanceMeters ? (totalDistanceMeters / 1000).toFixed(2) : '0.00'} km
            </Text>
          ) : null}
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>
            Capture: {lastCaptureTime} ms
          </Text>
          <Text style={styles.performanceText}>
            Inference: {metrics.inferenceTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Preprocess: {metrics.preprocessingTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Total: {metrics.totalLatencyMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            YOLO: {yoloInferenceTime.toFixed(0)} ms
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>
            Frames: {frameCount}
          </Text>
          <Text style={styles.performanceText}>
            Objects: {yoloDetections.length}
          </Text>
          <Text style={styles.performanceText}>
            Predictions: {predictionCount}
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.debugText} numberOfLines={2}>
            {debugStatus}
          </Text>
        </View>

        {/* Back Button (Top-Right) */}
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <Text style={styles.backButtonText}>✕</Text>
        </TouchableOpacity>

        {/* Status Indicator */}
        <View style={styles.statusIndicator}>
          <View style={[
            styles.statusDot,
            { backgroundColor: (isCameraReady && isCapturing) ? '#00ff00' : '#666666' }
          ]} />
          <Text style={styles.statusText}>
            {!isCameraReady ? 'Camera initializing...' : isCapturing ? 'Capturing' : 'Paused'}
          </Text>
          {(!isModelLoaded || !isYOLOModelLoaded) && (
            <Text style={styles.statusText}> | Demo Mode</Text>
          )}
        </View>

        {/* Direction Label (Bottom) */}
        <View style={styles.directionContainer}>
          <Text style={styles.directionLabel}>{directionLabel}</Text>
          {currentPrediction && (
            <Text style={styles.confidenceText}>
              {(confidence * 100).toFixed(1)}%
            </Text>
          )}
        </View>

        {/* Frame Buffer Progress */}
        <View style={styles.bufferProgress}>
          <View style={styles.bufferContainer}>
            <View 
              style={[
                styles.bufferFill,
                { width: `${(frameBufferRef.current.getFrameCount() / SEQ_LEN) * 100}%` }
              ]} 
            />
          </View>
          <Text style={styles.bufferText}>
            Buffer: {frameBufferRef.current.getFrameCount()}/{SEQ_LEN}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  
  camera: {
    flex: 1,
  },

  // Overlay container - positioned absolutely on top of camera
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    pointerEvents: 'box-none',
    zIndex: 100, // UI elements on top of everything
  },

  // Camera Status Overlay (Center)
  cameraStatusOverlay: {
    position: 'absolute',
    top: '40%',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    padding: 30,
    marginHorizontal: 40,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#444444',
  },
  cameraStatusText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '400',
    marginBottom: 8,
  },
  cameraStatusSubtext: {
    fontSize: 14,
    color: '#888888',
  },

  // Performance Overlay (Top-Left)
  performanceOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: 10,
    borderRadius: 4,
    minWidth: 120,
    borderWidth: 1,
    borderColor: '#333333',
  },
  performanceTitle: {
    fontSize: 10,
    fontWeight: '500',
    color: '#888888',
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  performanceText: {
    fontSize: 11,
    color: '#ffffff',
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  debugText: {
    fontSize: 9,
    color: '#00ff00',
    fontFamily: 'monospace',
    lineHeight: 12,
    maxWidth: 140,
  },
  performanceDivider: {
    height: 1,
    backgroundColor: '#333333',
    marginVertical: 4,
  },

  // Back Button (Top-Right)
  backButton: {
    position: 'absolute',
    top: 50,
    right: 16,
    width: 40,
    height: 40,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#333333',
  },
  backButtonText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '300',
  },

  // Status Indicator
  statusIndicator: {
    position: 'absolute',
    top: 56,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  statusText: {
    fontSize: 11,
    color: '#ffffff',
    fontWeight: '400',
  },

  // Direction Label (Bottom)
  directionContainer: {
    position: 'absolute',
    bottom: 80,
    left: 20,
    right: 20,
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderWidth: 1,
    borderColor: '#333333',
  },
  directionLabel: {
    fontSize: 40,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 6,
  },
  confidenceText: {
    fontSize: 12,
    color: '#888888',
    marginTop: 6,
  },

  // Buffer Progress
  bufferProgress: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  bufferContainer: {
    width: '100%',
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 1,
    overflow: 'hidden',
  },
  bufferFill: {
    height: '100%',
    backgroundColor: '#ffffff',
    borderRadius: 1,
  },
  bufferText: {
    fontSize: 10,
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: 4,
  },

  // Permission States
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  permissionText: {
    fontSize: 16,
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 20,
  },
  permissionButtonText: {
    fontSize: 14,
    color: '#000000',
    fontWeight: '500',
  },
});
