import * as Speech from 'expo-speech';
import { Detection } from './yoloInference';

/**
 * Configuration options for object-to-speech announcements.
 *
 * The service applies these values to candidate filtering,
 * announcement pacing, and speech playback settings.
 */
export interface ObjectSpeechServiceOptions {
  /** Minimum confidence required for a detection to be considered. */
  confidenceThreshold?: number;
  /** Cooldown for repeating announcements of the same class. */
  sameClassCooldownMs?: number;
  /** Cooldown applied between all announcements. */
  globalCooldownMs?: number;
  /** Minimum priority increase required to interrupt active speech. */
  interruptPriorityDelta?: number;
  /** Text-to-speech playback rate. */
  rate?: number;
  /** Text-to-speech playback pitch. */
  pitch?: number;
  /** BCP-47 language code passed to the speech engine. */
  language?: string;
}

type DirectionLabel = 'left' | 'ahead' | 'right';

/**
 * Internal representation of a detection candidate that can be spoken.
 */
interface AnnouncementCandidate {
  className: string;
  confidence: number;
  area: number;
  danger: number;
  direction: DirectionLabel;
  priority: number;
  message: string;
}

interface ActiveSpeechState {
  className: string;
  priority: number;
}

const AREA_EPSILON = 0.0001;
const CONFIDENCE_EPSILON = 0.0001;

const DEFAULT_OPTIONS: Required<ObjectSpeechServiceOptions> = {
  confidenceThreshold: 0.45,
  sameClassCooldownMs: 4000,
  globalCooldownMs: 1200,
  interruptPriorityDelta: 0.2,
  rate: 0.98,
  pitch: 1.0,
  language: 'en-US',
};

const DANGER_WEIGHTS: Record<string, number> = {
  person: 0.85,
  bicycle: 0.9,
  motorcycle: 0.98,
  car: 1.0,
  bus: 1.0,
  truck: 1.0,
  train: 1.0,
  dog: 0.65,
};

/**
 * Announces object detections using speech output.
 *
 * The service speaks a single, highest-priority object per call.
 * For multi-object frames, it prioritizes the nearest object by
 * using the largest normalized bounding-box area, with confidence
 * and danger as tie-breakers for stable and consistent output.
 */
export class ObjectSpeechService {
  private readonly options: Required<ObjectSpeechServiceOptions>;
  private readonly lastClassAnnouncementMs = new Map<string, number>();
  private lastAnnouncementMs = 0;
  private activeSpeech: ActiveSpeechState | null = null;
  private activeSpeechToken = 0;
  private enabled = true;

  constructor(options?: ObjectSpeechServiceOptions) {
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
    };
  }

  /** Enables or disables speech announcements globally. */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;

    if (!enabled) {
      void Speech.stop();
      this.activeSpeech = null;
      this.activeSpeechToken += 1;
    }
  }

  /**
   * Announces one object candidate from the current detection list.
   *
   * The method returns early when no valid candidate is available,
   * when cooldowns block output, or when active speech should not be interrupted.
   */
  async announceDetections(detections: Detection[]): Promise<void> {
    if (!this.enabled || detections.length === 0) {
      return;
    }

    const candidate = this.pickBestCandidate(detections);
    if (!candidate) {
      return;
    }

    const now = Date.now();
    if (!this.passesCooldowns(candidate, now)) {
      return;
    }

    const isSpeaking = await Speech.isSpeakingAsync().catch(() => false);
    if (isSpeaking) {
      if (!this.shouldInterrupt(candidate)) {
        return;
      }

      await Speech.stop().catch(() => undefined);
    }

    this.lastClassAnnouncementMs.set(candidate.className, now);
    this.lastAnnouncementMs = now;
    this.activeSpeech = {
      className: candidate.className,
      priority: candidate.priority,
    };
    const token = ++this.activeSpeechToken;

    Speech.speak(candidate.message, {
      language: this.options.language,
      rate: this.options.rate,
      pitch: this.options.pitch,
      onDone: () => {
        this.clearActiveSpeech(token);
      },
      onStopped: () => {
        this.clearActiveSpeech(token);
      },
      onError: () => {
        this.clearActiveSpeech(token);
      },
    });
  }

  /** Stops current speech playback and clears active speech state. */
  async stop(): Promise<void> {
    await Speech.stop().catch(() => undefined);
    this.activeSpeechToken += 1;
    this.activeSpeech = null;
  }

  /** Disposes service state and resets announcement history. */
  async dispose(): Promise<void> {
    await this.stop();
    this.lastClassAnnouncementMs.clear();
    this.lastAnnouncementMs = 0;
  }

  private clearActiveSpeech(token: number): void {
    if (this.activeSpeechToken === token) {
      this.activeSpeech = null;
    }
  }

  private pickBestCandidate(detections: Detection[]): AnnouncementCandidate | null {
    let bestCandidate: AnnouncementCandidate | null = null;

    for (const detection of detections) {
      const confidence = detection.confidence;

      // Confidence must be strictly greater than the configured threshold.
      if (!Number.isFinite(confidence) || confidence <= this.options.confidenceThreshold) {
        continue;
      }

      const xMin = detection.boundingBox.x;
      const width = detection.boundingBox.width;
      const height = detection.boundingBox.height;
      const xMax = xMin + width;

      if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || !Number.isFinite(width) || !Number.isFinite(height)) {
        continue;
      }

      if (width <= 0 || height <= 0) {
        continue;
      }

      const area = width * height;
      const danger = this.getDangerWeight(detection.className);
      const direction = this.getDirectionFromBounds(xMin, xMax);

      // Priority emphasizes proximity first so the nearest/largest object is preferred.
      const priority = (Math.min(1, area) * 0.75) + (confidence * 0.2) + (danger * 0.05);
      const spokenClass = this.toSpokenClassName(detection.className);
      const message = direction === 'ahead'
        ? `${spokenClass} ahead`
        : `${spokenClass} on the ${direction}`;

      const candidate: AnnouncementCandidate = {
        className: detection.className,
        confidence,
        area,
        danger,
        direction,
        priority,
        message,
      };

      if (!bestCandidate || this.isBetterCandidate(candidate, bestCandidate)) {
        bestCandidate = candidate;
      }
    }

    return bestCandidate;
  }

  private isBetterCandidate(
    candidate: AnnouncementCandidate,
    currentBest: AnnouncementCandidate,
  ): boolean {
    if (candidate.area > currentBest.area + AREA_EPSILON) {
      return true;
    }

    if (Math.abs(candidate.area - currentBest.area) <= AREA_EPSILON) {
      if (candidate.confidence > currentBest.confidence + CONFIDENCE_EPSILON) {
        return true;
      }

      if (Math.abs(candidate.confidence - currentBest.confidence) <= CONFIDENCE_EPSILON) {
        return candidate.danger > currentBest.danger;
      }
    }

    return false;
  }

  private passesCooldowns(candidate: AnnouncementCandidate, now: number): boolean {
    const lastForClass = this.lastClassAnnouncementMs.get(candidate.className) ?? 0;
    const sameClassCooldownPassed = (now - lastForClass) >= this.options.sameClassCooldownMs;
    if (!sameClassCooldownPassed) {
      return false;
    }

    const globalCooldownPassed = (now - this.lastAnnouncementMs) >= this.options.globalCooldownMs;
    if (!globalCooldownPassed && !this.shouldInterrupt(candidate)) {
      return false;
    }

    return true;
  }

  private shouldInterrupt(candidate: AnnouncementCandidate): boolean {
    if (!this.activeSpeech) {
      return true;
    }

    // Repeated class announcements do not interrupt active speech.
    if (candidate.className === this.activeSpeech.className) {
      return false;
    }

    return candidate.priority >= (this.activeSpeech.priority + this.options.interruptPriorityDelta);
  }

  private getDirectionFromBounds(xMin: number, xMax: number): DirectionLabel {
    const clampedMin = Math.max(0, Math.min(1, xMin));
    const clampedMax = Math.max(0, Math.min(1, xMax));
    const centerX = (clampedMin + clampedMax) * 0.5;

    if (centerX <= 0.33) {
      return 'left';
    }

    if (centerX >= 0.67) {
      return 'right';
    }

    return 'ahead';
  }

  private getDangerWeight(className: string): number {
    const normalized = className.trim().toLowerCase();
    return DANGER_WEIGHTS[normalized] ?? 0.6;
  }

  private toSpokenClassName(rawClassName: string): string {
    const cleaned = rawClassName
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (!cleaned) {
      return 'Object';
    }

    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
  }
}
