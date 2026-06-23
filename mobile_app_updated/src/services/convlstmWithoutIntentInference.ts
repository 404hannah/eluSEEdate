/**
 * TFLite Inference Service - ConvLSTM Without Intent
 * 
 * Handles model loading and inference for ConvLSTM turn prediction (no intent channels)
 * Uses react-native-fast-tflite for efficient on-device inference
 * 
 * NOTE: Requires a development build (not Expo Go) for native TFLite support
 * Run: npx expo prebuild && npx expo run:android
 */

import { CLASS_NAMES, ClassId, PredictionClass } from '../config/modelConfig';
import { ProcessedTensor } from './preprocessor';

// TFLite import - requires development build
let loadTensorflowModel: any = null;

// Track if we're in demo mode (Expo Go) or real mode (dev build)
let isDemoMode = true;

const EXPECTED_CONVLSTM_INPUT_SHAPE = [1, 20, 6, 128, 128] as const;
const EXPECTED_CONVLSTM_INPUT_ELEMENTS =
  EXPECTED_CONVLSTM_INPUT_SHAPE[0] *
  EXPECTED_CONVLSTM_INPUT_SHAPE[1] *
  EXPECTED_CONVLSTM_INPUT_SHAPE[2] *
  EXPECTED_CONVLSTM_INPUT_SHAPE[3] *
  EXPECTED_CONVLSTM_INPUT_SHAPE[4];

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  // react-native-fast-tflite must be loaded dynamically at runtime.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  isDemoMode = false;
  console.log('[ConvLSTM-TFLite] react-native-fast-tflite loaded successfully');
} catch {
  console.log('[ConvLSTM-TFLite] react-native-fast-tflite not available (Expo Go mode)');
  console.log('[ConvLSTM-TFLite] Real inference unavailable without native TFLite support');
  isDemoMode = true;
}

/**
 * Prediction result from model inference
 */
export interface PredictionResult {
  classId: ClassId;           // Predicted class (0, 1, 2)
  className: PredictionClass; // Human-readable class name
  confidence: number;         // Prediction confidence (0-1)
  probabilities: number[];    // All class probabilities
  inferenceTimeMs: number;    // Time taken for inference
}

/**
 * Performance metrics for tracking
 */
export interface PerformanceMetrics {
  preprocessingTimeMs: number;
  inferenceTimeMs: number;
  totalLatencyMs: number;
  fps: number;
}

/**
 * TFLite Model Manager
 * Handles loading and running inference with the ConvLSTM model
 */
class TFLiteModelManager {
  private isLoaded: boolean = false;
  private model: any = null;
  private demoMode: boolean = isDemoMode;

  /**
   * Load the TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    // Check if we're in demo mode (Expo Go)
    if (this.demoMode || !loadTensorflowModel) {
      console.log('[ConvLSTM-TFLite] â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•');
      console.log('[ConvLSTM-TFLite] âš ï¸  TFLite INFERENCE UNAVAILABLE');
      console.log('[ConvLSTM-TFLite] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€');
      console.log('[ConvLSTM-TFLite] Camera and UI work, but no model inference will run');
      console.log('[ConvLSTM-TFLite] ');
      console.log('[ConvLSTM-TFLite] To use TFLite inference, create a dev build:');
      console.log('[ConvLSTM-TFLite]   1. npx expo prebuild');
      console.log('[ConvLSTM-TFLite]   2. npx expo run:android');
      console.log('[ConvLSTM-TFLite] â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•');
      
      this.isLoaded = false;
      this.demoMode = true;
      return false;
    }

    try {
      console.log('[ConvLSTM-TFLite] Loading ConvLSTM model from assets...');
      
      // Load model from bundled assets with GPU delegate enabled
      // The model is in assets/model/convlstm.tflite (float16 optimized)
      const modelOptions = {
        // Enable GPU delegate for hardware acceleration
        // Falls back to CPU if GPU not available
        useGpu: true,
      };
      
      this.model = await loadTensorflowModel(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('../../assets/model/convlstm.tflite'),
        modelOptions
      );
      
      this.isLoaded = true;
      this.demoMode = false;
      console.log('[ConvLSTM-TFLite] âœ… Model loaded successfully with GPU acceleration!');
      console.log('[ConvLSTM-TFLite] Model: float16 with Global Average Pooling');
      console.log('[ConvLSTM-TFLite] Model ready for real-time inference');
      
      // Warm up with dummy inference
      console.log('[ConvLSTM-TFLite] Warming up model...');
      await this.warmUp();
      console.log('[ConvLSTM-TFLite] Model warm-up complete');
      
      return true;
    } catch (error: any) {
      console.error('[ConvLSTM-TFLite] âŒ Failed to load model:', error?.message || error);
      console.log('[ConvLSTM-TFLite] Model not available for inference');
      this.demoMode = true;
      return false;
    }
  }

  /**
   * Check if model is loaded (real inference available)
   */
  isModelLoaded(): boolean {
    return this.isLoaded && !this.demoMode;
  }

  /**
   * Check if running in demo mode
   */
  isInDemoMode(): boolean {
    return this.demoMode;
  }

  /**
   * Validate ConvLSTM input tensor shape before inference.
   * Prevents silent regressions when preprocessing layout changes.
   */
  private validateInputTensor(tensor: ProcessedTensor): void {
    const rawShape = Array.isArray(tensor.shape) ? tensor.shape : [];

    if (rawShape.length !== EXPECTED_CONVLSTM_INPUT_SHAPE.length) {
      throw new Error(
        `Invalid ConvLSTM input rank: expected ${EXPECTED_CONVLSTM_INPUT_SHAPE.length}, got ${rawShape.length}`
      );
    }

    const normalizedShape = rawShape.map((value) => Number(value));

    for (let index = 0; index < EXPECTED_CONVLSTM_INPUT_SHAPE.length; index += 1) {
      if (normalizedShape[index] !== EXPECTED_CONVLSTM_INPUT_SHAPE[index]) {
        throw new Error(
          `Invalid ConvLSTM input shape: expected [${EXPECTED_CONVLSTM_INPUT_SHAPE.join(', ')}], got [${normalizedShape.join(', ')}]`
        );
      }
    }

    if (tensor.data.length !== EXPECTED_CONVLSTM_INPUT_ELEMENTS) {
      throw new Error(
        `Invalid ConvLSTM input length: expected ${EXPECTED_CONVLSTM_INPUT_ELEMENTS}, got ${tensor.data.length}`
      );
    }
  }

  /**
   * Run inference on preprocessed tensor
   * 
   * @param tensor - Preprocessed frame sequence tensor
   * @returns Prediction result with class and confidence
   */
  async runInference(tensor: ProcessedTensor): Promise<PredictionResult> {
    const startTime = performance.now();

    try {
      if (this.demoMode || !this.isLoaded || !this.model) {
        throw new Error('ConvLSTM model is not loaded');
      }

      this.validateInputTensor(tensor);

      // Run model inference
      // Input: Float32Array with shape [1, 20, 6, 128, 128]
      const outputTensor = await this.model.run([tensor.data]);

      // Get output (should be [1, 3] for 3 classes)
      const output = Array.from(outputTensor[0] as ArrayLike<number>);

      if (output.length !== CLASS_NAMES.length) {
        console.warn(
          `[ConvLSTM-TFLite] Unexpected output size: expected ${CLASS_NAMES.length}, got ${output.length}`
        );
      }
      
      const inferenceTimeMs = performance.now() - startTime;

      // Apply softmax to get probabilities
      const probabilities = this.softmax(output);
      
      // Get predicted class
      const classId = this.argmax(probabilities) as ClassId;
      const className = CLASS_NAMES[classId] as PredictionClass;
      const confidence = probabilities[classId];
      
      return {
        classId,
        className,
        confidence,
        probabilities,
        inferenceTimeMs
      };
    } catch (error: any) {
      console.error('[ConvLSTM-TFLite] Inference failed:', error?.message || error);
      throw new Error(error?.message || 'ConvLSTM inference failed');
    }
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (this.demoMode || !this.model) return;
    
    try {
      const dummyData = new Float32Array(1 * 20 * 6 * 128 * 128);
      await this.model.run([dummyData]);
      console.log('[ConvLSTM-TFLite] Warm-up successful');
    } catch (error) {
      console.warn('[ConvLSTM-TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Softmax activation function
   */
  private softmax(logits: number[]): number[] {
    const sanitizedLogits = logits.map((value) => (Number.isFinite(value) ? value : 0));

    if (sanitizedLogits.some((value, index) => value !== logits[index])) {
      console.warn('[ConvLSTM-TFLite] Non-finite logits detected; replaced with 0');
    }

    const maxLogit = Math.max(...sanitizedLogits);
    const expValues = sanitizedLogits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);

    if (!Number.isFinite(sumExp) || sumExp <= 0) {
      const uniform = 1 / Math.max(1, expValues.length);
      console.warn('[ConvLSTM-TFLite] Invalid softmax denominator; returning uniform probabilities');
      return expValues.map(() => uniform);
    }

    return expValues.map(x => x / sumExp);
  }

  /**
   * Argmax - find index of maximum value
   */
  private argmax(arr: number[]): number {
    return arr.reduce((maxIdx, val, idx, array) => 
      val > array[maxIdx] ? idx : maxIdx, 0);
  }

  /**
   * Unload model and free resources
   */
  async unloadModel(): Promise<void> {
    if (this.model) {
      // TFLite models don't have explicit dispose, just null the reference
      this.model = null;
      console.log('[ConvLSTM-TFLite] Model unloaded');
    }
    this.isLoaded = false;
  }
}

/**
 * Singleton model manager instance
 */
let modelManager: TFLiteModelManager | null = null;

export function getModelManager(): TFLiteModelManager {
  if (!modelManager) {
    modelManager = new TFLiteModelManager();
  }
  return modelManager;
}

/**
 * High-level inference function
 */
export async function runPrediction(tensor: ProcessedTensor): Promise<{
  prediction: PredictionResult;
  metrics: PerformanceMetrics;
}> {
  const manager = getModelManager();
  const prediction = await manager.runInference(tensor);

  const metrics: PerformanceMetrics = {
    preprocessingTimeMs: tensor.processingTimeMs,
    inferenceTimeMs: prediction.inferenceTimeMs,
    totalLatencyMs: tensor.processingTimeMs + prediction.inferenceTimeMs,
    fps: 1000 / (tensor.processingTimeMs + prediction.inferenceTimeMs)
  };

  return { prediction, metrics };
}

/**
 * Initialize the model (call on app startup)
 */
export async function initializeModel(): Promise<boolean> {
  const manager = getModelManager();
  return manager.loadModel();
}

/**
 * Cleanup model resources (call on app close)
 */
export async function cleanupModel(): Promise<void> {
  const manager = getModelManager();
  await manager.unloadModel();
}

/**
 * Check if running in demo mode
 */
export function isRunningInDemoMode(): boolean {
  return getModelManager().isInDemoMode();
}

