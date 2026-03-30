/**
 * Image Utilities
 * 
 * Helper functions for decoding and processing camera images
 * Uses real JPEG decoding so models receive true camera pixels.
 */

import { FRAME_WIDTH, FRAME_HEIGHT } from '../config/modelConfig';

const jpeg: any = require('jpeg-js');

/**
 * Decode a base64 image to raw RGBA pixel data
 * 
 * Strategy:
 * - Decode JPEG bytes with jpeg-js
 * - Resize to target dimensions while preserving real pixel values
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
): Promise<{ data: Uint8Array; width: number; height: number }> {
  try {
    // Remove data URI prefix if present
    let base64Data = base64Image;
    if (base64Image.startsWith('data:')) {
      base64Data = base64Image.split(',')[1];
    }
    
    const pixelData = decodeAndResizeJpeg(base64Data, targetWidth, targetHeight);
    
    return {
      data: pixelData,
      width: targetWidth,
      height: targetHeight
    };
  } catch (error: any) {
    console.error('[ImageUtils] Failed to decode base64 image:', error?.message || error);
    throw error;
  }
}

/**
 * Decode JPEG bytes to RGBA and resize with nearest-neighbor.
 * This keeps preprocessing fast while preserving real scene content.
 */
function decodeAndResizeJpeg(
  base64: string,
  width: number,
  height: number
): Uint8Array {
  const binaryString = atob(base64);
  const jpegBytes = new Uint8Array(binaryString.length);

  for (let i = 0; i < binaryString.length; i++) {
    jpegBytes[i] = binaryString.charCodeAt(i);
  }

  const decoded = jpeg.decode(jpegBytes, { useTArray: true });
  if (!decoded?.data || !decoded.width || !decoded.height) {
    throw new Error('JPEG decode returned empty pixel data');
  }

  const srcData: Uint8Array = decoded.data;
  const srcWidth: number = decoded.width;
  const srcHeight: number = decoded.height;

  if (srcWidth === width && srcHeight === height) {
    return srcData;
  }

  const out = new Uint8Array(width * height * 4);
  const scaleX = srcWidth / width;
  const scaleY = srcHeight / height;

  for (let y = 0; y < height; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), srcHeight - 1);
    for (let x = 0; x < width; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), srcWidth - 1);
      const srcIdx = (srcY * srcWidth + srcX) * 4;
      const dstIdx = (y * width + x) * 4;

      out[dstIdx] = srcData[srcIdx];
      out[dstIdx + 1] = srcData[srcIdx + 1];
      out[dstIdx + 2] = srcData[srcIdx + 2];
      out[dstIdx + 3] = srcData[srcIdx + 3];
    }
  }

  return out;
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
