import { useCallback, useEffect, useRef, useState } from 'react';
import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition';
import * as Vosk from 'react-native-vosk';

interface SpeakMessageOptions {
  message: string;
  language?: string;
  statusWhileSpeaking?: string;
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
}

type ListenerWithRemove = { remove: () => void };

export function useVoiceInteraction(options?: UseVoiceInteractionOptions) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [readyToListen, setReadyToListen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState(options?.initialVoiceStatus ?? 'Initializing...');

  const defaultLanguage = options?.defaultLanguage ?? 'en-US';
  const defaultListeningDelayMs = options?.listeningDelayMs ?? 1000;

  const readyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingListeningRef = useRef<TransitionToListeningOptions | null>(null);
  const voskResultListenerRef = useRef<ListenerWithRemove | null>(null);
  const expoListenersRef = useRef<ListenerWithRemove[]>([]);
  const pingSoundRef = useRef<Audio.Sound | null>(null);

  const clearReadyTimer = useCallback(() => {
    if (readyTimerRef.current) {
      clearTimeout(readyTimerRef.current);
      readyTimerRef.current = null;
    }
  }, []);

  const stopVoskListening = useCallback(async () => {
    if (voskResultListenerRef.current) {
      voskResultListenerRef.current.remove();
      voskResultListenerRef.current = null;
    }

    try {
      await Vosk.stop();
    } catch {
      // Ignore stop failures during cleanup.
    }

    setIsListening(false);
  }, []);

  const stopExpoListening = useCallback(async () => {
    expoListenersRef.current.forEach((listener) => {
      listener.remove();
    });
    expoListenersRef.current = [];

    try {
      await Promise.resolve(ExpoSpeechRecognitionModule.abort());
    } catch {
      // Ignore stop failures during cleanup.
    }

    setIsListening(false);
  }, []);

  const emitListeningCue = useCallback(async () => {
    const hapticPromise = Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    const pingPromise = (async () => {
      if (!pingSoundRef.current) {
        return;
      }
      await pingSoundRef.current.replayAsync();
    })();

    await Promise.allSettled([hapticPromise, pingPromise]);
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

    try {
      Speech.speak(speakOptions.message, {
        language: speakOptions.language ?? defaultLanguage,
        onStart: () => {
          setIsSpeaking(true);
        },
        onDone: () => {
          setIsSpeaking(false);
          if (speakOptions.onDone) {
            speakOptions.onDone();
          }
        },
        onStopped: () => {
          setIsSpeaking(false);
        },
        onError: () => {
          setIsSpeaking(false);
          if (speakOptions.onError) {
            speakOptions.onError();
          }
        },
      });
    } catch {
      setIsSpeaking(false);
      if (speakOptions.onError) {
        speakOptions.onError();
      }
    }
  }, [clearReadyTimer, defaultLanguage]);

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

  const skipSpeech = useCallback(async () => {
    clearReadyTimer();

    try {
      await Speech.stop();
    } catch {
      // Ignore stop failures during user skip.
    }

    const pendingListening = pendingListeningRef.current;
    await transitionToListening(
      pendingListening ?? {
        statusWhileListening: voiceStatus,
        emitCue: true,
      }
    );
  }, [clearReadyTimer, transitionToListening, voiceStatus]);

  const startVoskListening = useCallback(async (listeningOptions: VoskListeningOptions) => {
    await stopVoskListening();

    try {
      await Vosk.start({ grammar: listeningOptions.grammar });

      voskResultListenerRef.current = Vosk.onResult((result: string) => {
        listeningOptions.onResult(result);
      }) as ListenerWithRemove;

      setIsListening(true);

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }

      return true;
    } catch (error: any) {
      setIsListening(false);

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

    try {
      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!permission.granted) {
        setVoiceStatus('Microphone permission denied');
        return false;
      }

      const resultListener = ExpoSpeechRecognitionModule.addListener('result', (event) => {
        if (listeningOptions.onResult) {
          void listeningOptions.onResult(event);
        }
      });

      const endListener = ExpoSpeechRecognitionModule.addListener('end', () => {
        setIsListening(false);
        if (listeningOptions.onEnd) {
          void listeningOptions.onEnd();
        }
      });

      const errorListener = ExpoSpeechRecognitionModule.addListener('error', (event) => {
        setIsListening(false);
        if (listeningOptions.onError) {
          listeningOptions.onError(event);
        }
      });

      expoListenersRef.current = [resultListener, endListener, errorListener];

      ExpoSpeechRecognitionModule.start(
        listeningOptions.startOptions ?? {
          lang: defaultLanguage,
          interimResults: false,
          continuous: false,
        }
      );

      setIsListening(true);

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }

      return true;
    } catch {
      await stopExpoListening();
      setIsListening(false);
      setVoiceStatus('Voice command disabled');
      return false;
    }
  }, [defaultLanguage, stopExpoListening]);

  const stopAllVoiceActivity = useCallback(async () => {
    clearReadyTimer();
    pendingListeningRef.current = null;
    setReadyToListen(false);

    await Promise.allSettled([
      stopVoskListening(),
      stopExpoListening(),
      Speech.stop(),
    ]);

    setIsSpeaking(false);
  }, [clearReadyTimer, stopExpoListening, stopVoskListening]);

  useEffect(() => {
    let isMounted = true;

    const loadCueSound = async () => {
      try {
        const { sound } = await Audio.Sound.createAsync(
          // This asset is expected to exist for assistive earcon playback.
          require('../../assets/sounds/ping.wav'),
          { shouldPlay: false, volume: 1.0 }
        );

        if (!isMounted) {
          await sound.unloadAsync();
          return;
        }

        pingSoundRef.current = sound;
      } catch {
        pingSoundRef.current = null;
      }
    };

    void loadCueSound();

    return () => {
      isMounted = false;
      clearReadyTimer();
      pendingListeningRef.current = null;
      void stopAllVoiceActivity();

      if (pingSoundRef.current) {
        void pingSoundRef.current.unloadAsync();
        pingSoundRef.current = null;
      }
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
