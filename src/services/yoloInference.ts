/**
 * YOLO Inference Service
 * 
 * Handles model loading and inference for YOLOv12 object detection
 * Uses react-native-fast-tflite for efficient on-device inference
 * 
 * NOTE: Requires a development build (not Expo Go) for native TFLite support
 * Run: npx expo prebuild && npx expo run:android
 */

import { YOLO_CLASS_NAMES } from '../config/modelConfig';
import { FrameData } from './preprocessor';

const ALLOWED_CLASS_IDS = new Set<number>([0, 1, 2, 3, 5, 7, 9, 11, 13, 14, 15, 16, 39]);

// TFLite import - requires development build
let loadTensorflowModel: any = null;

// Track if we're in demo mode (Expo Go) or real mode (dev build)
let isDemoMode = true;

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  isDemoMode = false;
} catch (e) {
  isDemoMode = true;
}

/**
 * Bounding box for detected object
 */
export interface BoundingBox {
  x: number;      // Top-left x coordinate (normalized 0-1)
  y: number;      // Top-left y coordinate (normalized 0-1)
  width: number;  // Width (normalized 0-1)
  height: number; // Height (normalized 0-1)
}

/**
 * Single object detection result
 */
export interface Detection {
  classId: number;           // YOLO class ID
  className: string;         // Human-readable class name
  confidence: number;        // Detection confidence (0-1)
  boundingBox: BoundingBox;  // Object bounding box
}

/**
 * YOLO detection result from model inference
 */
export interface YOLOResult {
  detections: Detection[];    // List of detected objects
  inferenceTimeMs: number;    // Time taken for inference
  frameWidth: number;         // Input frame width
  frameHeight: number;        // Input frame height
}

/**
 * YOLO Model Manager
 * Handles loading and running inference with the YOLOv12 model
 */
class YOLOModelManager {
  private isLoaded: boolean = false;
  private model: any = null;
  private demoMode: boolean = isDemoMode;
  private confidenceThreshold: number = 0.25; // Minimum confidence to report detection

  /**
   * Load the YOLO TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    // Check if we're in demo mode (Expo Go)
    if (this.demoMode || !loadTensorflowModel) {
      console.warn('[YOLO-TFLite] Inference unavailable: running in demo mode.');
      this.isLoaded = false;
      this.demoMode = true;
      return false;
    }

    try {
      // Load model from bundled assets with GPU delegate enabled
      const modelOptions = {
        useGpu: true, // Enable GPU acceleration
      };

      this.model = await loadTensorflowModel(
        require('../../assets/model/yolo.tflite'),
        modelOptions
      );

      this.isLoaded = true;
      this.demoMode = false;

      // Warm up with dummy inference
      await this.warmUp();

      return true;
    } catch (error: any) {
      console.error('[YOLO-TFLite] ❌ Failed to load model:', error?.message || error);
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
   * Set confidence threshold for detections
   */
  setConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
  }

  /**
   * Run YOLO inference on a single frame
   * 
   * @param frame - Single frame data from camera
   * @returns YOLO detection result with bounding boxes
   */
  async runInference(frame: FrameData): Promise<YOLOResult> {
    const startTime = performance.now();

    try {
      if (this.demoMode || !this.isLoaded || !this.model) {
        throw new Error('YOLO model is not loaded');
      }

      // Preprocess frame for YOLO (resize to model input size, normalize)
      const preprocessed = this.preprocessFrame(frame);
      
      // Run model inference
      const outputTensor = await this.model.run([preprocessed.data]);

      // Parse YOLO output (format depends on the active YOLOv12 model).
      const detections = this.parseYOLOOutput(outputTensor, frame.width, frame.height);

      const inferenceTimeMs = performance.now() - startTime;

      return {
        detections,
        inferenceTimeMs,
        frameWidth: frame.width,
        frameHeight: frame.height
      };
    } catch (error: any) {
      console.error('[YOLO-TFLite] Inference failed:', error?.message || error);
      throw new Error(error?.message || 'YOLO inference failed');
    }
  }

  /**
   * Preprocess frame for YOLO input
   * Input shape is flattened RGB data for a 128x128 frame.
   * Pixels are written in channel-last order (R, G, B per pixel).
   * Converts RGBA camera data to RGB, resizes, and normalizes to [0, 1]
   */
  private preprocessFrame(frame: FrameData): { data: Float32Array; width: number; height: number } {
    const inputSize = 128;
    const channels = 3;
    
    // Flattened channel-last tensor buffer.
    const data = new Float32Array(1 * channels * inputSize * inputSize);
    
    // Calculate scaling factors
    const scaleX = frame.width / inputSize;
    const scaleY = frame.height / inputSize;
    
    // Resize and convert RGBA to RGB in channel-last format.
    for (let y = 0; y < inputSize; y++) {
      for (let x = 0; x < inputSize; x++) {
        // Map to original frame coordinates (nearest neighbor)
        const srcX = Math.min(Math.floor(x * scaleX), frame.width - 1);
        const srcY = Math.min(Math.floor(y * scaleY), frame.height - 1);
        const srcIdx = (srcY * frame.width + srcX) * 4; // RGBA = 4 bytes per pixel

        // Output index in channel-last format
        const dstIdx = (y * inputSize + x) * 3;

        // Extract and normalize RGB values (0-255 -> 0-1)
        const r = frame.data[srcIdx] / 255.0;
        const g = frame.data[srcIdx + 1] / 255.0;
        const b = frame.data[srcIdx + 2] / 255.0;

        // Store in channel-last format
        data[0 + dstIdx] = r; // R channel
        data[1 + dstIdx] = g; // G channel
        data[2 + dstIdx] = b; // B channel
      }
    }
    
    return {
      data,
      width: inputSize,
      height: inputSize
    };
  }

  /**
   * Parse YOLO model output into detection objects
   * Output shape: (1, 84, 336) where 84 = [4 bbox coords + 80 class scores]
   * 336 = number of potential detections from various anchor boxes/scales
   */
  private parseYOLOOutput(outputTensor: any, frameWidth: number, frameHeight: number): Detection[] {
    const detections: Detection[] = [];
    
    try {
      // Output shape: (1, 84, 336)
      // - 84 values per detection: [x, y, w, h, class_scores...80]
      // - 336 potential detections
      const numDetections = 336;
      const numClasses = 80;
      const bboxCoords = 4;
      
      // Access the output data (format depends on TFLite binding)
      const outputData = outputTensor[0] || outputTensor;
      
      // Detect tensor layout: (1, 84, 336) vs (1, 336, 84)
      const totalValues = outputData?.length || 0;
      
      // Infer the layout based on typical YOLO patterns.
      // If length matches, the layout is determined as [84, 336] or [336, 84].
      let isTransposed = true; // Assume [84, 336] by default
      
      // Check whether data resembles the [336, 84] format.
      // In [336, 84] format, each detection contains 84 consecutive values.
      // Value-range comparison provides a heuristic for selecting the likely format.
      if (totalValues >= 84 * 2) {
        const val0 = outputData[0];
        const val84 = outputData[84];
        const val1 = outputData[1];
        const val336 = outputData[336];
        
        // If values at stride=84 are more similar than values at stride=336,
        // it's likely [336, 84] format
        if (Math.abs(val0 - val84) < Math.abs(val0 - val336)) {
          isTransposed = false;
        }
      }
      
      // Parse each detection
      for (let i = 0; i < numDetections; i++) {
        // Extract bbox coordinates (x, y, w, h)
        // Handle both transposed [84, 336] and non-transposed [336, 84] layouts
        let x, y, w, h;
        
        if (isTransposed) {
          // Transposed format: [84, 336] - each channel is contiguous
          x = outputData[0 * numDetections + i];      // Center X
          y = outputData[1 * numDetections + i];      // Center Y
          w = outputData[2 * numDetections + i];      // Width
          h = outputData[3 * numDetections + i];      // Height
        } else {
          // Non-transposed format: [336, 84] - each detection is contiguous
          x = outputData[i * 84 + 0];
          y = outputData[i * 84 + 1];
          w = outputData[i * 84 + 2];
          h = outputData[i * 84 + 3];
        }
        
        // Find class with highest confidence
        let maxClassScore = -Infinity;
        let maxClassId = 0;
        
        for (let c = 0; c < numClasses; c++) {
          let classScore;
          
          if (isTransposed) {
            // Transposed format: [84, 336]
            classScore = outputData[(bboxCoords + c) * numDetections + i];
          } else {
            // Non-transposed format: [336, 84]
            classScore = outputData[i * 84 + bboxCoords + c];
          }
          
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            maxClassId = c;
          }
        }
        
        // Model outputs probabilities directly (no sigmoid needed)
        // Float16 model outputs are already floating values.
        const confidence = Math.max(0, Math.min(1, maxClassScore));
        
        // Filter by confidence threshold (use actual threshold, not temp one)
        if (confidence >= this.confidenceThreshold && ALLOWED_CLASS_IDS.has(maxClassId)) {
          // Convert from center format (x, y, w, h) to corner format (x, y, width, height)
          const boxX = Math.max(0, Math.min(1, x - w / 2)); // Top-left X
          const boxY = Math.max(0, Math.min(1, y - h / 2)); // Top-left Y
          const boxW = Math.max(0, Math.min(1, w));         // Width
          const boxH = Math.max(0, Math.min(1, h));         // Height
          
          // Filter out invalid or tiny boxes (likely false positives)
          const MIN_BOX_SIZE = 0.01; // Minimum 1% of image size
          if (boxW > MIN_BOX_SIZE && boxH > MIN_BOX_SIZE && 
              boxX >= 0 && boxY >= 0 && 
              (boxX + boxW) <= 1 && (boxY + boxH) <= 1) {
            
            detections.push({
              classId: maxClassId,
              className: YOLO_CLASS_NAMES[maxClassId] || `class_${maxClassId}`,
              confidence: confidence,
              boundingBox: {
                x: boxX,
                y: boxY,
                width: boxW,
                height: boxH
              }
            });
          }
        }
      }
      
      // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
      const nmsDetections = this.applyNMS(detections, 0.45); // IoU threshold = 0.45
      
      return nmsDetections;
    } catch (error: any) {
      console.error('[YOLO-TFLite] Error parsing output:', error?.message || error);
      return [];
    }
  }

  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   */
  private calculateIoU(boxA: BoundingBox, boxB: BoundingBox): number {
    // Calculate intersection area
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
    const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
    
    const intersectionWidth = Math.max(0, xB - xA);
    const intersectionHeight = Math.max(0, yB - yA);
    const intersectionArea = intersectionWidth * intersectionHeight;
    
    // Calculate union area
    const boxAArea = boxA.width * boxA.height;
    const boxBArea = boxB.width * boxB.height;
    const unionArea = boxAArea + boxBArea - intersectionArea;
    
    // Return IoU
    return unionArea > 0 ? intersectionArea / unionArea : 0;
  }

  /**
   * Apply Non-Maximum Suppression (NMS) to remove overlapping detections
   */
  private applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) return [];
    
    // Sort detections by confidence (highest first)
    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    
    const keep: Detection[] = [];
    const suppressed = new Set<number>();
    
    for (let i = 0; i < sorted.length; i++) {
      if (suppressed.has(i)) continue;
      
      const currentBox = sorted[i];
      keep.push(currentBox);
      
      // Suppress overlapping boxes of the same class
      for (let j = i + 1; j < sorted.length; j++) {
        if (suppressed.has(j)) continue;
        
        const compareBox = sorted[j];
        
        // Only compare boxes of the same class
        if (currentBox.classId === compareBox.classId) {
          const iou = this.calculateIoU(currentBox.boundingBox, compareBox.boundingBox);
          
          if (iou > iouThreshold) {
            suppressed.add(j);
          }
        }
      }
    }
    
    return keep;
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (this.demoMode || !this.model) return;
    
    try {
      const dummyFrame: FrameData = {
        data: new Uint8Array(128 * 128 * 4),
        width: 128,
        height: 128,
        timestamp: Date.now()
      };
      
      await this.runInference(dummyFrame);
    } catch (error) {
      console.warn('[YOLO-TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Unload model and free resources
   */
  async unloadModel(): Promise<void> {
    if (this.model) {
      this.model = null;
    }
    this.isLoaded = false;
  }
}

/**
 * Singleton model manager instance
 */
let yoloModelManager: YOLOModelManager | null = null;

export function getYOLOModelManager(): YOLOModelManager {
  if (!yoloModelManager) {
    yoloModelManager = new YOLOModelManager();
  }
  return yoloModelManager;
}

/**
 * High-level YOLO detection function
 */
export async function runYOLODetection(frame: FrameData): Promise<YOLOResult> {
  const manager = getYOLOModelManager();
  return manager.runInference(frame);
}

/**
 * Initialize the YOLO model (call on app startup)
 */
export async function initializeYOLOModel(): Promise<boolean> {
  const manager = getYOLOModelManager();
  return manager.loadModel();
}

/**
 * Cleanup YOLO model resources (call on app close)
 */
export async function cleanupYOLOModel(): Promise<void> {
  const manager = getYOLOModelManager();
  await manager.unloadModel();
}

/**
 * Check if YOLO is running in demo mode
 */
export function isYOLOInDemoMode(): boolean {
  return getYOLOModelManager().isInDemoMode();
}

/**
 * Set YOLO confidence threshold
 */
export function setYOLOConfidenceThreshold(threshold: number): void {
  getYOLOModelManager().setConfidenceThreshold(threshold);
}
