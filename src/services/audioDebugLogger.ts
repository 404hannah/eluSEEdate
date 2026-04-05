const AUDIO_DEBUG_PREFIX = '[AUDIO-DEBUG]';

export type VoiceListenerState = 'Active' | 'Inactive' | 'Error';

function normalizePhrase(phrase: string): string {
  return phrase.replace(/\s+/g, ' ').trim();
}

export function logAudioDebugTtsStart(phrase: string): void {
  console.log(`${AUDIO_DEBUG_PREFIX} TTS Start: ${normalizePhrase(phrase)}`);
}

export function logAudioDebugTtsFinished(phrase: string): void {
  console.log(`${AUDIO_DEBUG_PREFIX} TTS Finished: ${normalizePhrase(phrase)}`);
}

export function logAudioDebugEarconTriggered(soundName: string): void {
  console.log(`${AUDIO_DEBUG_PREFIX} Earcon Triggered: ${soundName}`);
}

export function logAudioDebugVoiceListenerStatus(
  status: VoiceListenerState,
  details?: string,
): void {
  const suffix = details ? ` (${details})` : '';

  if (status === 'Error') {
    console.error(`${AUDIO_DEBUG_PREFIX} Voice Listener Status: ${status}${suffix}`);
    return;
  }

  console.log(`${AUDIO_DEBUG_PREFIX} Voice Listener Status: ${status}${suffix}`);
}
