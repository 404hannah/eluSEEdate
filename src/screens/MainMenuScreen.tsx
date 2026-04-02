/**
 * Main Menu Screen
 * 
 * Entry point of the app with a Start button
 * Navigates to CameraScreen when pressed or by saying "Start"
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Platform,
  BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/types';
import * as Vosk from 'react-native-vosk';
import * as Speech from 'expo-speech';

// Module-level flag to ensure TTS only speaks once per app session
let hasSpokenGreeting = false;

type MainMenuScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'MainMenu'>;
};

const { width } = Dimensions.get('window');

export default function MainMenuScreen({ navigation }: MainMenuScreenProps) {
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('Initializing...');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [readyToListen, setReadyToListen] = useState(false); // New state
  const resultListenerRef = useRef<any>(null);
  const hasNavigatedRef = useRef(false);


  const handleStartPress = () => {
    Speech.speak('Starting', { language: 'en-US' });
    navigation.navigate('Choice');
  };

  const handleExitPress = () => {
    Speech.speak('Exiting', { language: 'en-US' });
    // For mobile apps, exiting is not always supported, but we can try:
    if (Platform.OS === 'android') {
      // eslint-disable-next-line no-undef
      BackHandler.exitApp();
    }
  };

  // Speak startup greeting once per app session, then start listening after a delay
  useEffect(() => {
    if (!hasSpokenGreeting) {
      hasSpokenGreeting = true;
      Speech.speak(
        'Starting EluSEEdate. You can say Start to begin the app or Exit to exit the app.',
        {
          language: 'en-US',
          onDone: () => {
            setTimeout(() => {
              setReadyToListen(true);
            }, 1000); // 1 second delay after TTS
          },
        }
      );
    } else {
      setReadyToListen(true);
    }
  }, []);

  // Load Vosk model on component mount
  useEffect(() => {
    let isMounted = true;

    const loadModel = async () => {
      try {
        setVoiceStatus('Loading voice model...');
        await Vosk.loadModel('model-en-us');
        if (isMounted) {
          setModelLoaded(true);
          setVoiceStatus('Say "Start" to begin');
        }
      } catch (error) {
        console.error('Failed to load Vosk model:', error);
        if (isMounted) {
          setVoiceStatus('Voice command disabled');
        }
      }
    };

    loadModel();

    return () => {
      isMounted = false;
    };
  }, []);

  // Start/stop voice recognition when screen is focused/unfocused
  useFocusEffect(
    useCallback(() => {
      hasNavigatedRef.current = false;
      let listeningTimeout: ReturnType<typeof setTimeout> | null = null;

      const startListening = async () => {
        if (!modelLoaded || !readyToListen) return;
        try {
          await Vosk.start({ grammar: ['start', 'exit', '[unk]'] });
          setIsListening(true);
          setVoiceStatus('Say "Start" to begin');
          resultListenerRef.current = Vosk.onResult((result: string) => {
            console.log('Voice result:', result);
            const lowerResult = result.toLowerCase();
            if (!hasNavigatedRef.current) {
              if (lowerResult.includes('start')) {
                hasNavigatedRef.current = true;
                setVoiceStatus('Starting...');
                Speech.speak('Starting', { language: 'en-US' });
                Vosk.stop();
                setIsListening(false);
                navigation.navigate('Choice');
              } else if (lowerResult.includes('exit')) {
                hasNavigatedRef.current = true;
                setVoiceStatus('Exiting...');
                Speech.speak('Exiting', { language: 'en-US' });
                Vosk.stop();
                setIsListening(false);
                if (Platform.OS === 'android') {
                  BackHandler.exitApp();
                }
              }
            }
          });
        } catch (error: any) {
          console.error('Failed to start voice recognition:', error);
          if (error?.message?.includes('permission') || error?.message?.includes('Permission')) {
            setVoiceStatus('Microphone permission denied');
          } else {
            setVoiceStatus('Voice command disabled');
          }
          setIsListening(false);
        }
      };

      // Only start listening if readyToListen is true
      if (readyToListen) {
        listeningTimeout = setTimeout(startListening, 0);
      }

      // Cleanup when screen loses focus
      return () => {
        if (resultListenerRef.current) {
          resultListenerRef.current.remove();
          resultListenerRef.current = null;
        }
        Vosk.stop();
        setIsListening(false);
        if (listeningTimeout) clearTimeout(listeningTimeout);
      };
    }, [modelLoaded, navigation, readyToListen])
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      {/* Header Section */}
      <View style={styles.headerSection}>
        <Text style={styles.title}>EluSEEdate</Text>
        <Text style={styles.subtitle}>Turn Prediction</Text>
        <Text style={styles.version}>v1.0.4</Text>
      </View>

      {/* Center Section with Start and Exit Buttons */}
      <View style={styles.centerSection}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleStartPress}
          activeOpacity={0.7}
        >
          <Text style={styles.startButtonText}>Start</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.startButton, { marginTop: 20, backgroundColor: '#222' }]}
          onPress={handleExitPress}
          activeOpacity={0.7}
        >
          <Text style={[styles.startButtonText, { color: '#fff' }]}>Exit</Text>
        </TouchableOpacity>
        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>
      </View>

      {/* Footer Section */}
      <View style={styles.footerSection}>
        <Text style={styles.footerText}>
          Point camera at the road
        </Text>
        
        {/* Debug Logs Button */}
        <TouchableOpacity
          style={styles.debugButton}
          onPress={() => navigation.navigate('Logs')}
          activeOpacity={0.7}
        >
          <Text style={styles.debugButtonText}>📋 Debug Logs</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  
  // Header Section
  headerSection: {
    flex: 2,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  title: {
    fontSize: 36,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 4,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '300',
    color: '#888888',
    marginTop: 8,
  },
  version: {
    fontSize: 12,
    color: '#444444',
    marginTop: 16,
  },

  // Center Section
  centerSection: {
    flex: 3,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  startButton: {
    width: width * 0.5,
    height: 60,
    backgroundColor: '#ffffff',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  startButtonText: {
    fontSize: 20,
    fontWeight: '500',
    color: '#000000',
    letterSpacing: 2,
  },
  
  // Voice Status
  voiceStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 24,
  },
  voiceIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#444444',
    marginRight: 8,
  },
  voiceIndicatorActive: {
    backgroundColor: '#00ff00',
  },
  voiceStatusText: {
    fontSize: 12,
    color: '#666666',
  },

  // Footer Section
  footerSection: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  footerText: {
    fontSize: 12,
    color: '#666666',    marginBottom: 20,
  },
  debugButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  debugButtonText: {
    fontSize: 14,
    color: '#888888',  },
});
