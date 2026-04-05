import { useCallback, useEffect, useRef, useState } from 'react';
import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition';
import * as Vosk from 'react-native-vosk';
import {
  logAudioDebugEarconTriggered,
  logAudioDebugTtsFinished,
  logAudioDebugTtsStart,
  logAudioDebugVoiceListenerStatus,
} from '../services/audioDebugLogger';

interface SpeakMessageOptions {
  message: string;
  language?: string;
  statusWhileSpeaking?: string;
  playEarcon?: boolean;
  onDone?: () => void;
  onError?: () => void;
}

interface TransitionToListeningOptions {
  statusWhileListening?: string;
  onListeningReady?: () => void | Promise<void>;
  emitCue?: boolean;
}

interface SpeakThenListenOptions extends SpeakMessageOptions, TransitionToListeningOptions {
  delayMs?: number;
}

interface VoskListeningOptions {
  grammar: string[];
  statusWhileListening?: string;
  onResult: (result: string) => void;
  onErrorMessage?: string;
}

interface ExpoListeningOptions {
  startOptions?: {
    lang?: string;
    interimResults?: boolean;
    continuous?: boolean;
    contextualStrings?: string[];
    androidIntentOptions?: Record<string, string>;
    iosTaskHint?: 'unspecified' | 'dictation' | 'search' | 'confirmation';
  };
  statusWhileListening?: string;
  onResult?: (event: { isFinal?: boolean; results?: { transcript?: string }[] }) => void | Promise<void>;
  onEnd?: () => void | Promise<void>;
  onError?: (event: { error?: string; message?: string }) => void;
}

interface UseVoiceInteractionOptions {
  initialVoiceStatus?: string;
  defaultLanguage?: string;
  listeningDelayMs?: number;
  ttsEarconEnabled?: boolean;
}

type ListenerWithRemove = { remove: () => void };
type ListenerEngine = 'vosk' | 'expo';

let voiceInteractionInstanceCounter = 0;
let speechOwnerId: number | null = null;
let voskOwnerId: number | null = null;
let expoOwnerId: number | null = null;
let globalVoskResultListener: ListenerWithRemove | null = null;
let globalExpoListeners: ListenerWithRemove[] = [];
let pingSoundSingleton: Audio.Sound | null = null;
let pingSoundLoader: Promise<Audio.Sound | null> | null = null;

function removeListenerSafe(listener: ListenerWithRemove | null | undefined): void {
  if (!listener) {
    return;
  }

  try {
    listener.remove();
  } catch {
    // Ignore listener cleanup errors.
  }
}

function clearGlobalExpoListeners(): void {
  globalExpoListeners.forEach((listener) => {
    removeListenerSafe(listener);
  });
  globalExpoListeners = [];
}

async function getPingSoundSingleton(): Promise<Audio.Sound | null> {
  if (pingSoundSingleton) {
    return pingSoundSingleton;
  }

  if (!pingSoundLoader) {
    pingSoundLoader = Audio.Sound.createAsync(
      require('../../assets/sounds/ping.wav'),
      { shouldPlay: false, volume: 1.0 },
    )
      .then(({ sound }) => {
        pingSoundSingleton = sound;
        return sound;
      })
      .catch(() => null);
  }

  return pingSoundLoader;
}

export function useVoiceInteraction(options?: UseVoiceInteractionOptions) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [readyToListen, setReadyToListen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState(options?.initialVoiceStatus ?? 'Initializing...');

  const defaultLanguage = options?.defaultLanguage ?? 'en-US';
  const defaultListeningDelayMs = options?.listeningDelayMs ?? 1000;
  const ttsEarconEnabled = options?.ttsEarconEnabled ?? true;
  const interactionIdRef = useRef(++voiceInteractionInstanceCounter);

  const readyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingListeningRef = useRef<TransitionToListeningOptions | null>(null);
  const voskResultListenerRef = useRef<ListenerWithRemove | null>(null);
  const expoListenersRef = useRef<ListenerWithRemove[]>([]);
  const activeListenerEngineRef = useRef<ListenerEngine | null>(null);
  const isVoskListeningRef = useRef(false);
  const isExpoListeningRef = useRef(false);

  const clearReadyTimer = useCallback(() => {
    if (readyTimerRef.current) {
      clearTimeout(readyTimerRef.current);
      readyTimerRef.current = null;
    }
  }, []);

  const playTtsEarcon = useCallback(async () => {
    const pingSound = await getPingSoundSingleton();
    if (!pingSound) {
      return;
    }

    try {
      logAudioDebugEarconTriggered('ping.wav');
      await pingSound.replayAsync();
    } catch {
      // Earcon failure should not block TTS.
    }
  }, []);

  const stopVoskListening = useCallback(async () => {
    const ownsVosk = voskOwnerId === interactionIdRef.current;

    if (voskResultListenerRef.current) {
      removeListenerSafe(voskResultListenerRef.current);

      if (globalVoskResultListener === voskResultListenerRef.current) {
        globalVoskResultListener = null;
      }

      voskResultListenerRef.current = null;
    }

    if (ownsVosk && globalVoskResultListener) {
      removeListenerSafe(globalVoskResultListener);
      globalVoskResultListener = null;
    }

    if (ownsVosk) {
      try {
        await Vosk.stop();
      } catch {
        // Ignore stop failures during cleanup.
      }

      voskOwnerId = null;
    }

    if (isVoskListeningRef.current) {
      logAudioDebugVoiceListenerStatus('Inactive', 'Vosk');
    }

    isVoskListeningRef.current = false;
    if (activeListenerEngineRef.current === 'vosk') {
      activeListenerEngineRef.current = null;
    }

    setIsListening(false);
  }, []);

  const stopExpoListening = useCallback(async () => {
    const ownsExpo = expoOwnerId === interactionIdRef.current;

    expoListenersRef.current.forEach((listener) => {
      removeListenerSafe(listener);
    });
    expoListenersRef.current = [];

    if (ownsExpo) {
      try {
        await Promise.resolve(ExpoSpeechRecognitionModule.abort());
      } catch {
        // Ignore stop failures during cleanup.
      }

      clearGlobalExpoListeners();
      expoOwnerId = null;
    }

    if (isExpoListeningRef.current) {
      logAudioDebugVoiceListenerStatus('Inactive', 'ExpoSpeechRecognition');
    }

    isExpoListeningRef.current = false;
    if (activeListenerEngineRef.current === 'expo') {
      activeListenerEngineRef.current = null;
    }

    setIsListening(false);
  }, []);

  const emitListeningCue = useCallback(async () => {
    await Promise.allSettled([
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium),
    ]);
  }, []);

  const transitionToListening = useCallback(async (listeningOptions?: TransitionToListeningOptions) => {
    clearReadyTimer();
    pendingListeningRef.current = null;

    setIsSpeaking(false);
    setReadyToListen(true);

    if (listeningOptions?.statusWhileListening) {
      setVoiceStatus(listeningOptions.statusWhileListening);
    }

    if (listeningOptions?.emitCue !== false) {
      await emitListeningCue();
    }

    if (listeningOptions?.onListeningReady) {
      await listeningOptions.onListeningReady();
    }
  }, [clearReadyTimer, emitListeningCue]);

  const speakMessage = useCallback((speakOptions: SpeakMessageOptions) => {
    clearReadyTimer();
    pendingListeningRef.current = null;
    setReadyToListen(false);

    if (speakOptions.statusWhileSpeaking) {
      setVoiceStatus(speakOptions.statusWhileSpeaking);
    }

    setIsSpeaking(true);

    const runSpeech = async () => {
      const previousSpeechOwnerId = speechOwnerId;
      speechOwnerId = interactionIdRef.current;

      if (
        previousSpeechOwnerId !== null
        && previousSpeechOwnerId !== interactionIdRef.current
      ) {
        try {
          await Speech.stop();
        } catch {
          // Ignore interruptions while ownership is transferred.
        }
      }

      if (ttsEarconEnabled && speakOptions.playEarcon !== false) {
        await playTtsEarcon();
      }

      let didFinalize = false;

      const finalizeSpeech = (callback?: () => void) => {
        if (speechOwnerId !== interactionIdRef.current) {
          return;
        }

        if (!didFinalize) {
          didFinalize = true;
          logAudioDebugTtsFinished(speakOptions.message);
        }

        speechOwnerId = null;
        setIsSpeaking(false);

        if (callback) {
          callback();
        }
      };

      try {
        logAudioDebugTtsStart(speakOptions.message);

        Speech.speak(speakOptions.message, {
          language: speakOptions.language ?? defaultLanguage,
          onStart: () => {
            if (speechOwnerId !== interactionIdRef.current) {
              return;
            }

            setIsSpeaking(true);
          },
          onDone: () => {
            finalizeSpeech(() => {
              if (speakOptions.onDone) {
                speakOptions.onDone();
              }
            });
          },
          onStopped: () => {
            finalizeSpeech();
          },
          onError: () => {
            if (speechOwnerId !== interactionIdRef.current) {
              return;
            }

            speechOwnerId = null;
            setIsSpeaking(false);

            if (speakOptions.onError) {
              speakOptions.onError();
            }
          },
        });
      } catch {
        if (speechOwnerId === interactionIdRef.current) {
          speechOwnerId = null;
        }

        setIsSpeaking(false);

        if (speakOptions.onError) {
          speakOptions.onError();
        }
      }
    };

    void runSpeech();
  }, [clearReadyTimer, defaultLanguage, playTtsEarcon, ttsEarconEnabled]);

  const speakThenListen = useCallback((speakOptions: SpeakThenListenOptions) => {
    const listeningOptions: TransitionToListeningOptions = {
      statusWhileListening: speakOptions.statusWhileListening,
      onListeningReady: speakOptions.onListeningReady,
      emitCue: true,
    };

    pendingListeningRef.current = listeningOptions;

    speakMessage({
      message: speakOptions.message,
      language: speakOptions.language,
      statusWhileSpeaking: speakOptions.statusWhileSpeaking,
      onDone: () => {
        const delayMs = Math.max(0, speakOptions.delayMs ?? defaultListeningDelayMs);
        clearReadyTimer();
        readyTimerRef.current = setTimeout(() => {
          void transitionToListening(pendingListeningRef.current ?? listeningOptions);
        }, delayMs);

        if (speakOptions.onDone) {
          speakOptions.onDone();
        }
      },
      onError: () => {
        void transitionToListening(pendingListeningRef.current ?? listeningOptions);
        if (speakOptions.onError) {
          speakOptions.onError();
        }
      },
    });
  }, [clearReadyTimer, defaultListeningDelayMs, speakMessage, transitionToListening]);

  const stopSpeechAndClearQueue = useCallback(async (reason: 'skip' | 'cleanup') => {
    if (speechOwnerId !== interactionIdRef.current) {
      return false;
    }

    try {
      console.log(`[AUDIO-DEBUG] TTS Interrupt: ${reason}`);
      await Promise.resolve(Speech.stop());
      // A second stop call helps flush queued utterances on some engines.
      await Promise.resolve(Speech.stop());
    } catch {
      // Ignore stop failures during interruption.
    }

    speechOwnerId = null;
    setIsSpeaking(false);
    return true;
  }, []);

  const skipSpeech = useCallback(async () => {
    clearReadyTimer();

    const interrupted = await stopSpeechAndClearQueue('skip');
    if (!interrupted) {
      try {
        await Promise.resolve(Speech.stop());
      } catch {
        // Ignore stop failures during user skip.
      }
    }

    const pendingListening = pendingListeningRef.current;
    await transitionToListening(
      pendingListening ?? {
        statusWhileListening: voiceStatus,
        emitCue: true,
      }
    );
  }, [clearReadyTimer, stopSpeechAndClearQueue, transitionToListening, voiceStatus]);

  const startVoskListening = useCallback(async (listeningOptions: VoskListeningOptions) => {
    await stopVoskListening();

    if (globalVoskResultListener) {
      removeListenerSafe(globalVoskResultListener);
      globalVoskResultListener = null;
    }

    if (voskOwnerId !== null && voskOwnerId !== interactionIdRef.current) {
      try {
        await Vosk.stop();
      } catch {
        // Ignore ownership handoff failures.
      }
    }

    voskOwnerId = interactionIdRef.current;

    try {
      await Vosk.start({ grammar: listeningOptions.grammar });

      voskResultListenerRef.current = Vosk.onResult((result: string) => {
        if (voskOwnerId !== interactionIdRef.current) {
          return;
        }

        listeningOptions.onResult(result);
      }) as ListenerWithRemove;

      globalVoskResultListener = voskResultListenerRef.current;
      activeListenerEngineRef.current = 'vosk';
      isVoskListeningRef.current = true;
      isExpoListeningRef.current = false;

      setIsListening(true);
      logAudioDebugVoiceListenerStatus('Active', 'Vosk');

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }

      return true;
    } catch (error: any) {
      if (voskOwnerId === interactionIdRef.current) {
        voskOwnerId = null;
      }

      activeListenerEngineRef.current = null;
      isVoskListeningRef.current = false;
      setIsListening(false);
      logAudioDebugVoiceListenerStatus(
        'Error',
        `Vosk${error?.message ? `: ${String(error.message)}` : ''}`,
      );

      if (listeningOptions.onErrorMessage) {
        setVoiceStatus(listeningOptions.onErrorMessage);
      } else if (error?.message?.toLowerCase?.().includes('permission')) {
        setVoiceStatus('Microphone permission denied');
      } else {
        setVoiceStatus('Voice command disabled');
      }

      return false;
    }
  }, [stopVoskListening]);

  const startExpoListening = useCallback(async (listeningOptions: ExpoListeningOptions) => {
    await stopExpoListening();

    if (expoOwnerId !== null && expoOwnerId !== interactionIdRef.current) {
      try {
        await Promise.resolve(ExpoSpeechRecognitionModule.abort());
      } catch {
        // Ignore ownership handoff failures.
      }

      clearGlobalExpoListeners();
    }

    expoOwnerId = interactionIdRef.current;

    try {
      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!permission.granted) {
        expoOwnerId = null;
        logAudioDebugVoiceListenerStatus('Error', 'ExpoSpeechRecognition: Permission denied');
        setVoiceStatus('Microphone permission denied');
        return false;
      }

      const resultListener = ExpoSpeechRecognitionModule.addListener('result', (event) => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

        if (listeningOptions.onResult) {
          void listeningOptions.onResult(event);
        }
      });

      const endListener = ExpoSpeechRecognitionModule.addListener('end', () => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

        expoOwnerId = null;
        clearGlobalExpoListeners();
        activeListenerEngineRef.current = null;
        if (isExpoListeningRef.current) {
          logAudioDebugVoiceListenerStatus('Inactive', 'ExpoSpeechRecognition');
        }

        isExpoListeningRef.current = false;
        setIsListening(false);

        if (listeningOptions.onEnd) {
          void listeningOptions.onEnd();
        }
      });

      const errorListener = ExpoSpeechRecognitionModule.addListener('error', (event) => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

        expoOwnerId = null;
        clearGlobalExpoListeners();
        activeListenerEngineRef.current = null;
        isExpoListeningRef.current = false;
        setIsListening(false);

        logAudioDebugVoiceListenerStatus(
          'Error',
          `ExpoSpeechRecognition${event.error ? `: ${String(event.error)}` : ''}`,
        );

        if (listeningOptions.onError) {
          listeningOptions.onError(event);
        }
      });

      expoListenersRef.current = [resultListener, endListener, errorListener];
      clearGlobalExpoListeners();
      globalExpoListeners = [...expoListenersRef.current];

      await Promise.resolve(ExpoSpeechRecognitionModule.start(
        listeningOptions.startOptions ?? {
          lang: defaultLanguage,
          interimResults: false,
          continuous: false,
        }
      ));

      activeListenerEngineRef.current = 'expo';
      isExpoListeningRef.current = true;
      isVoskListeningRef.current = false;
      setIsListening(true);
      logAudioDebugVoiceListenerStatus('Active', 'ExpoSpeechRecognition');

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }

      return true;
    } catch (error: any) {
      if (expoOwnerId === interactionIdRef.current) {
        expoOwnerId = null;
      }

      activeListenerEngineRef.current = null;
      isExpoListeningRef.current = false;
      await stopExpoListening();
      setIsListening(false);
      setVoiceStatus('Voice command disabled');

      logAudioDebugVoiceListenerStatus(
        'Error',
        `ExpoSpeechRecognition${error?.message ? `: ${String(error.message)}` : ''}`,
      );

      return false;
    }
  }, [defaultLanguage, stopExpoListening]);

  const stopAllVoiceActivity = useCallback(async () => {
    clearReadyTimer();
    pendingListeningRef.current = null;
    setReadyToListen(false);

    const ownsSpeech = speechOwnerId === interactionIdRef.current;

    await Promise.allSettled([
      stopVoskListening(),
      stopExpoListening(),
      ownsSpeech ? stopSpeechAndClearQueue('cleanup') : Promise.resolve(),
    ]);

    if (ownsSpeech) {
      speechOwnerId = null;
    }

    setIsSpeaking(false);
  }, [clearReadyTimer, stopExpoListening, stopSpeechAndClearQueue, stopVoskListening]);

  useEffect(() => {
    void getPingSoundSingleton();

    return () => {
      clearReadyTimer();
      pendingListeningRef.current = null;
      void stopAllVoiceActivity();
    };
  }, [clearReadyTimer, stopAllVoiceActivity]);

  return {
    isListening,
    isSpeaking,
    readyToListen,
    voiceStatus,
    setVoiceStatus,
    speakMessage,
    speakThenListen,
    transitionToListening,
    skipSpeech,
    startVoskListening,
    stopVoskListening,
    startExpoListening,
    stopExpoListening,
    stopAllVoiceActivity,
  };
}
