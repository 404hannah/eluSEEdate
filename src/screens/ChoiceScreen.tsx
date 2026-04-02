/**
 * Choice Screen
 * 
 * Allows users to choose between Wandering (NoIntent) and Destination (Intent) modes
 * Supports voice commands: "Wandering", "Destination", "Back"
 * 
 * Design: Minimalistic black & white (matches MainMenu)
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/types';
import * as Vosk from 'react-native-vosk';
import * as Speech from 'expo-speech';

type ChoiceScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Choice'>;
};

const { width } = Dimensions.get('window');

export default function ChoiceScreen({ navigation }: ChoiceScreenProps) {
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('Initializing...');
  const [readyToListen, setReadyToListen] = useState(false);
  const resultListenerRef = useRef<any>(null);
  const hasNavigatedRef = useRef(false);


  const handleWanderingPress = () => {
    Speech.speak('Starting wandering mode', {
      language: 'en-US',
      onDone: () => navigation.navigate('Wandering'),
    });
  };

  const handleDestinationPress = () => {
    Speech.speak('Opening wayfinding', {
      language: 'en-US',
      onDone: () => navigation.navigate('Wayfinding'),
    });
  };

  const handleBackPress = () => {
    Speech.speak('Going back', { language: 'en-US' });
    navigation.navigate('MainMenu');
  };

  // TTS greeting, then enable listening after it finishes + delay
  useFocusEffect(
    useCallback(() => {
      setReadyToListen(false);
      Speech.speak(
        'Choose your mode, Wandering or Destination. If you want to return to the main menu say back.',
        {
          language: 'en-US',
          onDone: () => {
            setTimeout(() => {
              setReadyToListen(true);
            }, 1000);
          },
        }
      );

      return () => {
        Speech.stop();
        setReadyToListen(false);
      };
    }, [])
  );

  // Start/stop voice recognition when screen is focused and ready
  useFocusEffect(
    useCallback(() => {
      hasNavigatedRef.current = false;
      let listeningTimeout: ReturnType<typeof setTimeout> | null = null;

      const startListening = async () => {
        if (!readyToListen) return;
        try {
          await Vosk.start({ grammar: ['wandering', 'destination', 'back', '[unk]'] });
          setIsListening(true);
          setVoiceStatus('Say "Wandering", "Destination", or "Back"');
          resultListenerRef.current = Vosk.onResult((result: string) => {
            console.log('Choice voice result:', result);
            const lowerResult = result.toLowerCase();
            if (!hasNavigatedRef.current) {
              if (lowerResult.includes('wandering')) {
                hasNavigatedRef.current = true;
                setVoiceStatus('Starting wandering mode...');
                Vosk.stop();
                setIsListening(false);
                Speech.speak('Starting wandering mode', {
                  language: 'en-US',
                  onDone: () => navigation.navigate('Wandering'),
                });
              } else if (lowerResult.includes('destination')) {
                hasNavigatedRef.current = true;
                setVoiceStatus('Opening wayfinding...');
                Vosk.stop();
                setIsListening(false);
                Speech.speak('Opening wayfinding', {
                  language: 'en-US',
                  onDone: () => navigation.navigate('Wayfinding'),
                });
              } else if (lowerResult.includes('back')) {
                hasNavigatedRef.current = true;
                setVoiceStatus('Going back...');
                Vosk.stop();
                setIsListening(false);
                Speech.speak('Going back', {
                  language: 'en-US',
                  onDone: () => navigation.navigate('MainMenu'),
                });
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

      if (readyToListen) {
        listeningTimeout = setTimeout(startListening, 0);
      }

      return () => {
        if (resultListenerRef.current) {
          resultListenerRef.current.remove();
          resultListenerRef.current = null;
        }
        Vosk.stop();
        setIsListening(false);
        if (listeningTimeout) clearTimeout(listeningTimeout);
      };
    }, [navigation, readyToListen])
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />

      {/* Header Section */}
      <View style={styles.headerSection}>
        <Text style={styles.title}>Choose Your Mode</Text>
      </View>

      {/* Center Section with mode buttons */}
      <View style={styles.centerSection}>
        <TouchableOpacity
          style={styles.modeButton}
          onPress={handleWanderingPress}
          activeOpacity={0.7}
        >
          <Text style={styles.modeButtonText}>Wandering</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20 }]}
          onPress={handleDestinationPress}
          activeOpacity={0.7}
        >
          <Text style={styles.modeButtonText}>Destination</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20, backgroundColor: '#222' }]}
          onPress={handleBackPress}
          activeOpacity={0.7}
        >
          <Text style={[styles.modeButtonText, { color: '#fff' }]}>Back</Text>
        </TouchableOpacity>

        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>
      </View>

      {/* Footer */}
      <View style={styles.footerSection} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  headerSection: {
    flex: 2,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  title: {
    fontSize: 30,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 3,
  },
  centerSection: {
    flex: 3,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  modeButton: {
    width: width * 0.5,
    height: 60,
    backgroundColor: '#ffffff',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modeButtonText: {
    fontSize: 20,
    fontWeight: '500',
    color: '#000000',
    letterSpacing: 2,
  },
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
  footerSection: {
    flex: 1,
  },
});
