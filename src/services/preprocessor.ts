/**
 * Video Preprocessor for Mobile ConvLSTM Turn Prediction
 * 
 * TypeScript port of the Python preprocessor
 * Prepares video frames for inference on mobile devices
 * 
 * Key Features:
 * - Processes frames at 10 FPS sampling rate
 * - Resizes to 128x128
 * - Normalizes pixel values to [0, 1]
 * - Adds intent channels (3 additional channels, all zeros for 'no intent')
 * - Returns tensor shape: [seq_len, channels, height, width] = [20, 6, 128, 128]
 */

import {
  SEQ_LEN,
  FPS,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  CHANNELS,
  DEVICE_CONFIG,
  ENABLE_INTENT_MODE
} from '../config/modelConfig';

type TensorChannelCount = 3 | 6;

const resolveTensorChannelCount = (channels: number): TensorChannelCount => {
  if (channels === 3 || channels === 6) {
    return channels;
  }

  throw new Error('VideoPreprocessor channels must be 3 or 6');
};

/**
 * Frame data structure
 * Holds raw pixel data from camera capture
 */
export interface FrameData {
  data: Uint8Array;         // Raw pixel data (RGBA format from camera)
  width: number;            // Original frame width
  height: number;           // Original frame height
  timestamp: number;        // Capture timestamp in ms
  sequenceId?: number;      // Monotonic capture sequence ID for debugging/tracing
  intent?: number;          // Intent: 0 - Front, 1 - Left, 2 - Right
  intentDistance: number;   // Distance before intent occurs
}

/**
 * Processed tensor ready for model inference
 */
export interface ProcessedTensor {
  data: Float32Array;    // Flattened tensor data
  shape: number[];       // Tensor shape [batch, seq_len, channels, height, width]
  processingTimeMs: number; // Time taken to preprocess
}

/**
 * Frame buffer configuration
 */
export interface FrameBufferConfig {
  maxFrames: number;     // Maximum frames to buffer
  samplingRate: number;  // Frame sampling rate (take every Nth frame)
  cameraFps: number;     // Camera's native FPS
}

/**
 * Circular buffer for managing frame sequence
 */
export class FrameBuffer {
  private frames: FrameData[] = [];
  private config: FrameBufferConfig;
  private frameCount: number = 0;

  constructor(cameraFps: number = DEVICE_CONFIG.cameraFps) {
    this.config = {
      maxFrames: SEQ_LEN,
      samplingRate: Math.max(1, Math.floor(cameraFps / FPS)),
      cameraFps: cameraFps
    };
  }

  /**
   * Add a frame to the buffer (with automatic sampling)
   * Returns true if frame was added, false if skipped due to sampling
   */
  addFrame(frame: FrameData): boolean {
    this.frameCount++;
    
    // Sample frames based on camera FPS
    if ((this.frameCount - 1) % this.config.samplingRate !== 0) {
      return false; // Skip this frame
    }

    // Add frame to buffer
    this.frames.push(frame);
    
    // Remove oldest frame if buffer exceeds max size
    if (this.frames.length > this.config.maxFrames) {
      this.frames.shift();
    }

    return true;
  }

  /**
   * Check if buffer has enough frames for inference
   */
  isReady(): boolean {
    return this.frames.length >= SEQ_LEN;
  }

  /**
   * Check if buffer has minimum frames for early prediction (with padding)
   * Early prediction available when at least 50% of required frames are collected
   */
  canPredictEarly(): boolean {
    const minFrames = Math.ceil(SEQ_LEN / 2);
    return this.frames.length >= minFrames;
  }

  /**
   * Get current frame count in buffer
   */
  getFrameCount(): number {
    return this.frames.length;
  }

  /**
   * Get all frames in buffer
   * If buffer not full, duplicates last frame to reach SEQ_LEN
   */
  getFrames(): FrameData[] {
    return [...this.frames];
  }

  /**
   * Get frames padded to SEQ_LEN by duplicating the last frame
   * Used for early predictions before buffer is full
   */
  getFramesPadded(): FrameData[] {
    const frames = [...this.frames];
    
    // Pad with duplicate of last frame if needed
    while (frames.length < SEQ_LEN) {
      frames.push(frames[frames.length - 1]);
    }
    
    return frames;
  }

  /**
   * Get bootstrap frames for the first prediction.
   *
   * Strategy:
   * - Take up to the latest half-sequence of unique frames.
   * - Duplicate each frame in order: 1,1,2,2,3,3...
   * - If still short, pad using the last frame.
   */
  getFramesBootstrapDoubled(): FrameData[] {
    if (this.frames.length === 0) {
      return [];
    }

    const maxUniqueFrames = Math.ceil(SEQ_LEN / 2);
    const sourceFrames = this.frames.slice(-maxUniqueFrames);
    const doubled: FrameData[] = [];

    for (const frame of sourceFrames) {
      doubled.push(frame);
      doubled.push(frame);
    }

    while (doubled.length < SEQ_LEN) {
      doubled.push(doubled[doubled.length - 1]);
    }

    return doubled.slice(0, SEQ_LEN);
  }

  /**
   * Clear the buffer
   */
  clear(): void {
    this.frames = [];
    this.frameCount = 0;
  }

  /**
   * Get buffer status
   */
  getStatus(): { current: number; required: number; ready: boolean } {
    return {
      current: this.frames.length,
      required: SEQ_LEN,
      ready: this.isReady()
    };
  }
}

/**
 * Video Preprocessor Class
 * Handles frame preprocessing for ConvLSTM model inference
 */
export class VideoPreprocessor {
  private height: number;
  private width: number;
  private seqLen: number;
  private normalize: boolean;
  private channels: TensorChannelCount;
  private framePlaneSize: number;
  private frameStride: number;
  private tensorBuffer: Float32Array;
  private outputShape: number[];

  constructor(
    height: number = FRAME_HEIGHT,
    width: number = FRAME_WIDTH,
    seqLen: number = SEQ_LEN,
    normalize: boolean = true,
    channels: number = CHANNELS
  ) {
    this.height = height;
    this.width = width;
    this.seqLen = seqLen;
    this.normalize = normalize;
    this.channels = resolveTensorChannelCount(channels);

    this.framePlaneSize = this.height * this.width;
    this.frameStride = this.channels * this.framePlaneSize;

    // Reused backing buffer: [1, seqLen, channels, height, width]
    this.tensorBuffer = new Float32Array(this.seqLen * this.frameStride);
    this.outputShape = [1, this.seqLen, this.channels, this.height, this.width];
  }

  /**
   * Preprocess a sequence of frames for model inference
   * 
   * Pipeline:
   * 1. Resize each frame to (height, width)
   * 2. Convert RGBA to RGB
   * 3. Normalize to [0, 1] if enabled
   * 4. Add intent channels (all zeros for 'no intent')
   * 5. Transpose to channels-first format
   * 6. Stack into sequence tensor
   * 
   * @param frames - Array of captured frames
   * @returns ProcessedTensor ready for model inference
   */
  preprocessFrameSequence(frames: FrameData[]): ProcessedTensor {
    const startTime = performance.now();

    if (frames.length !== this.seqLen) {
      throw new Error('Expected ' + this.seqLen + ' frames, got ' + frames.length);
    }

    // Process each frame
    for (let frameIdx = 0; frameIdx < this.seqLen; frameIdx++) {
      this.processFrame(frames[frameIdx], frameIdx, this.tensorBuffer);
    }

    const processingTimeMs = performance.now() - startTime;

    return {
      data: this.tensorBuffer,
      shape: this.outputShape,
      processingTimeMs
    };
  }

  /**
   * Process a single frame through the preprocessing pipeline
   * 
   * Steps:
   * 1. Resize to target dimensions
   * 2. Convert RGBA to RGB (camera captures RGBA)
   * 3. Normalize to [0, 1]
   * 4. Transpose to channels-first format
   * 5. Add intent channels (all zeros)
   */
  private processFrame(
    frame: FrameData,
    frameIdx: number,
    tensorData: Float32Array
  ): void {
    // Calculate offset in tensor for this frame
    // Tensor layout: [batch, seq, channels, height, width]
    // Batch index 0 is filled, so offset = frameIdx * channels * height * width.
    const frameOffset = frameIdx * this.frameStride;

    this.resizeNormalizeAndWriteFrame(frame, frameOffset, tensorData);

    // In 3-channel mode, only RGB values are packed.
    // In 6-channel mode, RGB is packed with intent slots reserved for addIntent writes.
  }

  /**
   * Resize, normalize, and write directly to final NCHW RGB tensor slots.
   * Uses nearest-neighbor interpolation for lower JS compute cost.
   */
  private resizeNormalizeAndWriteFrame(
    frame: FrameData,
    frameOffset: number,
    tensorData: Float32Array
  ): void {
    const frameWidth = frame.width;
    const frameHeight = frame.height;
    const source = frame.data;

    const maxSrcX = frameWidth - 1;
    const maxSrcY = frameHeight - 1;
    const channelOffset = [0, 0, 0, 0, 0, 0];

    for (let y = 0; y < this.height; y++) {
      const rowOffset = y * this.width;
      // Nearest-neighbor row lookup: target y -> source y.
      let srcY = Math.floor((y * frameHeight) / this.height);
      if (srcY > maxSrcY) {
        srcY = maxSrcY;
      }
      const srcRowBase = srcY * frameWidth;

      for (let x = 0; x < this.width; x++) {
        // Nearest-neighbor column lookup: target x -> source x.
        let srcX = Math.floor((x * frameWidth) / this.width);
        if (srcX > maxSrcX) {
          srcX = maxSrcX;
        }

        const srcBase = (srcRowBase + srcX) * 4;

        const pixelOffset = rowOffset + x;

        // Extract RGB from RGBA source.
        const valueR = source[srcBase];
        const valueG = source[srcBase + 1];
        const valueB = source[srcBase + 2];

        if (this.channels === 3) {
          const rgbBase = frameOffset + pixelOffset * 3;

          if (this.normalize) {
            tensorData[rgbBase] = valueR / 255.0;
            tensorData[rgbBase + 1] = valueG / 255.0;
            tensorData[rgbBase + 2] = valueB / 255.0;
          } else {
            tensorData[rgbBase] = valueR;
            tensorData[rgbBase + 1] = valueG;
            tensorData[rgbBase + 2] = valueB;
          }

          continue;
        }

        const packedBase = frameOffset + pixelOffset * 6;

        if (this.normalize) {
          tensorData[packedBase] = valueR / 255.0;
          tensorData[packedBase + 1] = valueG / 255.0;
          tensorData[packedBase + 2] = valueB / 255.0;
        } else {
          tensorData[packedBase] = valueR;
          tensorData[packedBase + 1] = valueG;
          tensorData[packedBase + 2] = valueB;
        }

        if (ENABLE_INTENT_MODE) {
          channelOffset[0] = packedBase;
          channelOffset[1] = packedBase + 1;
          channelOffset[2] = packedBase + 2;
          channelOffset[3] = packedBase + 3;
          channelOffset[4] = packedBase + 4;
          channelOffset[5] = packedBase + 5;

          this.addIntent(frame, tensorData, channelOffset, 0);
        }
      }
    }
  }

  private addIntent(
    frame: FrameData,
    tensorData: Float32Array,
    channelOffset: number[],
    pixelOffset: number
  ): void {
    // Obtain intent
    const intentClass = frame.intent || 0; // Default to front
    const intentDistance = frame.intentDistance || 0; // Default to 0

    // Check if intent would occur within the next few meters (<5)
    if (intentDistance <= 5) {
      // Fill the pixel's specified intent channel of intent (1)
      tensorData[channelOffset[intentClass + 3] + pixelOffset] = 1;
    } else {
      // Fill the pixel's front channel of intent (1)
      tensorData[channelOffset[3] + pixelOffset] = 1;
    }    
  }

  /**
   * Get expected output shape
   */
  getOutputShape(): number[] {
    return this.outputShape;
  }
}

/**
 * Singleton instance for easy access
 */
let preprocessorInstance: VideoPreprocessor | null = null;

export function getPreprocessor(): VideoPreprocessor {
  if (!preprocessorInstance) {
    preprocessorInstance = new VideoPreprocessor();
  }
  return preprocessorInstance;
}

/**
 * Convenience function for quick preprocessing
 */
export function preprocessFrames(frames: FrameData[]): ProcessedTensor {
  return getPreprocessor().preprocessFrameSequence(frames);
}
