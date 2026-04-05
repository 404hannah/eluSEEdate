/**
 * Logs Screen
 * Displays real-time console logs for debugging
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Modal,
  Pressable,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';

// Global log storage
interface LogEntry {
  id: number;
  timestamp: Date;
  level: 'log' | 'warn' | 'error' | 'info' | 'debug';
  message: string;
}

type LogFilter = 'all' | 'yolo' | 'convlstm' | 'audio' | 'errors';

interface LogFilterOption {
  key: LogFilter;
  label: string;
  description: string;
}

let logStorage: LogEntry[] = [];
let logIdCounter = 0;
let logListeners: ((logs: LogEntry[]) => void)[] = [];

const LOG_FILTER_OPTIONS: LogFilterOption[] = [
  {
    key: 'all',
    label: 'All Logs',
    description: 'Default view for all captured logs',
  },
  {
    key: 'yolo',
    label: 'YOLO',
    description: '[INFERENCE-DEBUG] and [PRIORITY-DEBUG]',
  },
  {
    key: 'convlstm',
    label: 'ConvLSTM',
    description: '[CONVLSTM-TRACE]',
  },
  {
    key: 'audio',
    label: 'Audio',
    description: '[AUDIO-DEBUG] and [AUDIO-TRACE]',
  },
  {
    key: 'errors',
    label: 'Errors',
    description: 'console.error entries and [ERROR] tags',
  },
];

// Override console methods to capture logs
const originalConsole = {
  log: console.log,
  warn: console.warn,
  error: console.error,
  info: console.info,
  debug: console.debug,
};

const captureLog = (level: LogEntry['level'], ...args: any[]) => {
  const message = args.map(arg => {
    if (typeof arg === 'object') {
      try {
        return JSON.stringify(arg, null, 2);
      } catch {
        return String(arg);
      }
    }
    return String(arg);
  }).join(' ');

  const entry: LogEntry = {
    id: logIdCounter++,
    timestamp: new Date(),
    level,
    message,
  };

  logStorage.push(entry);
  
  // Keep only last 500 logs to prevent memory issues
  if (logStorage.length > 500) {
    logStorage = logStorage.slice(-500);
  }

  // Notify listeners
  logListeners.forEach(listener => listener([...logStorage]));

  // Call original console method
  originalConsole[level](...args);
};

// Install console overrides
console.log = (...args) => captureLog('log', ...args);
console.warn = (...args) => captureLog('warn', ...args);
console.error = (...args) => captureLog('error', ...args);
console.info = (...args) => captureLog('info', ...args);
console.debug = (...args) => captureLog('debug', ...args);

export default function LogsScreen() {
  const navigation = useNavigation();
  const [logs, setLogs] = useState<LogEntry[]>([...logStorage]);
  const [filter, setFilter] = useState<LogFilter>('all');
  const [isFilterModalVisible, setIsFilterModalVisible] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    // Subscribe to log updates
    const listener = (newLogs: LogEntry[]) => {
      setLogs(newLogs);
      if (autoScroll) {
        setTimeout(() => {
          scrollViewRef.current?.scrollToEnd({ animated: true });
        }, 100);
      }
    };

    logListeners.push(listener);

    return () => {
      logListeners = logListeners.filter(l => l !== listener);
    };
  }, [autoScroll]);

  const getFilteredLogs = () => {
    switch (filter) {
      case 'yolo':
        return logs.filter(log => {
          const messageUpper = log.message.toUpperCase();
          return (
            messageUpper.includes('[INFERENCE-DEBUG]')
            || messageUpper.includes('[PRIORITY-DEBUG]')
          );
        });
      case 'convlstm':
        return logs.filter(log => log.message.toUpperCase().includes('[CONVLSTM-TRACE]'));
      case 'audio':
        return logs.filter(log => {
          const messageUpper = log.message.toUpperCase();
          return (
            messageUpper.includes('[AUDIO-DEBUG]')
            || messageUpper.includes('[AUDIO-TRACE]')
          );
        });
      case 'errors':
        return logs.filter(log => {
          const messageUpper = log.message.toUpperCase();
          return log.level === 'error' || messageUpper.includes('[ERROR]');
        });
      default:
        return logs;
    }
  };

  const selectedFilterOption = LOG_FILTER_OPTIONS.find(option => option.key === filter) ?? LOG_FILTER_OPTIONS[0];

  const clearLogs = () => {
    logStorage = [];
    setLogs([]);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'error': return '#ff4444';
      case 'warn': return '#ffaa00';
      case 'info': return '#44aaff';
      case 'debug': return '#aa44ff';
      default: return '#cccccc';
    }
  };

  const filteredLogs = getFilteredLogs();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Debug Logs</Text>
        <View style={{ width: 60 }} />
      </View>

      {/* Filter Dropdown */}
      <View style={styles.filterContainer}>
        <Text style={styles.filterLabel}>Category</Text>
        <TouchableOpacity
          style={styles.filterDropdownButton}
          onPress={() => setIsFilterModalVisible(true)}
          activeOpacity={0.8}
        >
          <View>
            <Text style={styles.filterDropdownTitle}>{selectedFilterOption.label}</Text>
            <Text style={styles.filterDropdownSubtitle}>{selectedFilterOption.description}</Text>
          </View>
          <Text style={styles.filterDropdownChevron}>▼</Text>
        </TouchableOpacity>
      </View>

      <Modal
        animationType="fade"
        transparent
        visible={isFilterModalVisible}
        onRequestClose={() => setIsFilterModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <Pressable
            style={styles.modalBackdrop}
            onPress={() => setIsFilterModalVisible(false)}
          />
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Log Category</Text>
            {LOG_FILTER_OPTIONS.map((option) => (
              <TouchableOpacity
                key={option.key}
                style={[
                  styles.modalOption,
                  filter === option.key && styles.modalOptionActive,
                ]}
                onPress={() => {
                  setFilter(option.key);
                  setIsFilterModalVisible(false);
                }}
                activeOpacity={0.85}
              >
                <Text style={styles.modalOptionLabel}>{option.label}</Text>
                <Text style={styles.modalOptionDescription}>{option.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Modal>

      {/* Action Buttons */}
      <View style={styles.actionContainer}>
        <TouchableOpacity 
          style={styles.actionButton}
          onPress={() => setAutoScroll(!autoScroll)}
        >
          <Text style={styles.actionButtonText}>
            Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.actionButton, styles.clearButton]}
          onPress={clearLogs}
        >
          <Text style={styles.actionButtonText}>Clear Logs</Text>
        </TouchableOpacity>
      </View>

      {/* Logs Display */}
      <ScrollView 
        ref={scrollViewRef}
        style={styles.logsContainer}
        contentContainerStyle={styles.logsContent}
      >
        {filteredLogs.length === 0 ? (
          <Text style={styles.emptyText}>No logs yet. Start using the app to see debug output.</Text>
        ) : (
          filteredLogs.map(log => (
            <View key={log.id} style={styles.logEntry}>
              <Text style={styles.logTime}>{formatTime(log.timestamp)}</Text>
              <Text style={[styles.logLevel, { color: getLevelColor(log.level) }]}>
                [{log.level.toUpperCase()}]
              </Text>
              <Text style={styles.logMessage}>{log.message}</Text>
            </View>
          ))
        )}
      </ScrollView>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Showing {filteredLogs.length} of {logs.length} logs
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  filterContainer: {
    paddingHorizontal: 10,
    paddingTop: 10,
    paddingBottom: 4,
  },
  filterLabel: {
    color: '#b5b5b5',
    fontSize: 12,
    marginBottom: 6,
    letterSpacing: 0.6,
  },
  filterDropdownButton: {
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: '#222',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  filterDropdownTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  filterDropdownSubtitle: {
    color: '#9b9b9b',
    fontSize: 11,
    marginTop: 2,
  },
  filterDropdownChevron: {
    color: '#d0d0d0',
    fontSize: 13,
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  modalCard: {
    backgroundColor: '#111111',
    borderWidth: 1,
    borderColor: '#2e2e2e',
    borderRadius: 12,
    padding: 12,
  },
  modalTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 10,
  },
  modalOption: {
    backgroundColor: '#1b1b1b',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#2c2c2c',
  },
  modalOptionActive: {
    borderColor: '#0a7ea4',
    backgroundColor: '#123845',
  },
  modalOptionLabel: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  modalOptionDescription: {
    color: '#9f9f9f',
    fontSize: 11,
    marginTop: 2,
  },
  actionContainer: {
    flexDirection: 'row',
    padding: 10,
    gap: 8,
  },
  actionButton: {
    flex: 1,
    paddingVertical: 10,
    backgroundColor: '#333',
    borderRadius: 8,
    alignItems: 'center',
  },
  clearButton: {
    backgroundColor: '#aa3333',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  logsContainer: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  logsContent: {
    padding: 10,
  },
  logEntry: {
    flexDirection: 'row',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#1a1a1a',
  },
  logTime: {
    color: '#666',
    fontSize: 11,
    fontFamily: 'monospace',
    marginRight: 8,
    minWidth: 90,
  },
  logLevel: {
    fontSize: 11,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginRight: 8,
    minWidth: 60,
  },
  logMessage: {
    flex: 1,
    color: '#ccc',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  emptyText: {
    color: '#666',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 40,
  },
  footer: {
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#333',
    alignItems: 'center',
  },
  footerText: {
    color: '#666',
    fontSize: 12,
  },
});
