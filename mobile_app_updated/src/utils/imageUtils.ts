/**
 * Image Utilities
 *
 * Helper functions for decoding and processing camera images.
 */

import { decode as decodeJpeg } from 'jpeg-js';
import * as ImageManipulator from 'expo-image-manipulator';
import { FRAME_WIDTH, FRAME_HEIGHT } from '../config/modelConfig';

interface PixelBufferResult {
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * Decode an image URI using native resize first, then decode tiny JPEG in JS.
 * This is much faster than decoding full camera-resolution JPEG in JS.
 */
export async function decodeImageUriToPixels(
  imageUri: string,
  targetWidth: number = FRAME_WIDTH,
  targetHeight: number = FRAME_HEIGHT
): Promise<PixelBufferResult> {
  try {
    const manipulated = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: targetWidth, height: targetHeight } }],
      {
        compress: 0.55,
        format: ImageManipulator.SaveFormat.JPEG,
        base64: true,
      }
    );

    if (!manipulated.base64) {
      throw new Error('ImageManipulator did not return base64 data');
    }

    const jpegBytes = base64ToUint8Array(manipulated.base64);
    const decoded = decodeJpeg(jpegBytes, { useTArray: true });

    if (!decoded?.data || decoded.width <= 0 || decoded.height <= 0) {
      throw new Error('JPEG decode returned empty image');
    }

    const resized = (decoded.width === targetWidth && decoded.height === targetHeight)
      ? decoded.data
      : resizeRgbaNearest(
          decoded.data,
          decoded.width,
          decoded.height,
          targetWidth,
          targetHeight
        );

    return {
      data: resized,
      width: targetWidth,
      height: targetHeight,
    };
  } catch (error: any) {
    console.error('[ImageUtils] Failed to decode image URI:', error?.message || error);
    return createGrayFallback(targetWidth, targetHeight);
  }
}

/**
 * Decode a base64 JPEG image to raw RGBA pixel data.
 * Uses jpeg-js so inference receives actual image pixels.
 *
 * @param base64Image - Base64 encoded image string (with or without data URI prefix)
 * @param targetWidth - Target width to resize to
 * @param targetHeight - Target height to resize to
 * @returns Promise<{ data: Uint8Array; width: number; height: number }>
 */
export async function decodeBase64ToPixels(
  base64Image: string,
  targetWidth: number = FRAME_WIDTH,
  targetHeight: number = FRAME_HEIGHT
): Promise<PixelBufferResult> {
  try {
    // Remove data URI prefix if present
    let base64Data = base64Image;
    if (base64Image.startsWith('data:')) {
      base64Data = base64Image.split(',')[1];
    }
    
    const jpegBytes = base64ToUint8Array(base64Data);
    const decoded = decodeJpeg(jpegBytes, { useTArray: true });

    if (!decoded?.data || decoded.width <= 0 || decoded.height <= 0) {
      throw new Error('JPEG decode returned empty image');
    }

    const resized = resizeRgbaNearest(
      decoded.data,
      decoded.width,
      decoded.height,
      targetWidth,
      targetHeight
    );
    
    return {
      data: resized,
      width: targetWidth,
      height: targetHeight
    };
  } catch (error: any) {
    console.error('[ImageUtils] Failed to decode base64 image:', error?.message || error);
    return createGrayFallback(targetWidth, targetHeight);
  }
}

function createGrayFallback(targetWidth: number, targetHeight: number): PixelBufferResult {
  const pixelCount = targetWidth * targetHeight;
  const fallbackData = new Uint8Array(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const value = 128;
    fallbackData[i * 4 + 0] = value;
    fallbackData[i * 4 + 1] = value;
    fallbackData[i * 4 + 2] = value;
    fallbackData[i * 4 + 3] = 255;
  }

  return {
    data: fallbackData,
    width: targetWidth,
    height: targetHeight,
  };
}

/**
 * Convert a base64 string to bytes.
 */
function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

/**
 * Resize RGBA frame data using nearest-neighbor interpolation.
 */
function resizeRgbaNearest(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array {
  const pixelCount = targetWidth * targetHeight;
  const rgbaData = new Uint8Array(pixelCount * 4);

  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), sourceWidth - 1);
      const srcY = Math.min(Math.floor(y * scaleY), sourceHeight - 1);

      const srcIdx = (srcY * sourceWidth + srcX) * 4;
      const dstIdx = (y * targetWidth + x) * 4;

      rgbaData[dstIdx] = source[srcIdx];
      rgbaData[dstIdx + 1] = source[srcIdx + 1];
      rgbaData[dstIdx + 2] = source[srcIdx + 2];
      rgbaData[dstIdx + 3] = source[srcIdx + 3];
    }
  }

  return rgbaData;
}

/**
 * Check if an image string is valid base64
 */
export function isValidBase64Image(base64: string): boolean {
  try {
    const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');
    atob(base64Data.substring(0, Math.min(100, base64Data.length)));
    return base64Data.length > 100; // Reasonable minimum length
  } catch {
    return false;
  }
}
