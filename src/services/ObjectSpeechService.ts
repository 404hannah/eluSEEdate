import * as Speech from 'expo-speech';
import { Detection } from './yoloInference';

export interface ObjectSpeechServiceOptions {
  confidenceThreshold?: number;
  sameClassCooldownMs?: number;
  globalCooldownMs?: number;
  interruptPriorityDelta?: number;
  rate?: number;
  pitch?: number;
  language?: string;
}

type DirectionLabel = 'left' | 'ahead' | 'right';

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

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;

    if (!enabled) {
      void Speech.stop();
      this.activeSpeech = null;
      this.activeSpeechToken += 1;
    }
  }

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

  async stop(): Promise<void> {
    await Speech.stop().catch(() => undefined);
    this.activeSpeechToken += 1;
    this.activeSpeech = null;
  }

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

      // Confidence must be strictly greater than 0.45.
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

      // Priority emphasizes danger first, then proximity (box area), then confidence.
      const priority = (danger * 0.55) + (Math.min(1, area) * 0.3) + (confidence * 0.15);
      const spokenClass = this.toSpokenClassName(detection.className);
      const message = direction === 'ahead'
        ? `${spokenClass} ahead`
        : `${spokenClass} on the ${direction}`;

      if (!bestCandidate || priority > bestCandidate.priority) {
        bestCandidate = {
          className: detection.className,
          confidence,
          area,
          danger,
          direction,
          priority,
          message,
        };
      }
    }

    return bestCandidate;
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
