/**
 * Logs Screen
 * Displays real-time console logs for debugging.
 *
 * Category filtering uses a modal dropdown selector to keep
 * diagnostics controls compact while preserving quick access.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  ListRenderItem,
  Modal,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
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
    description: 'No filter applied',
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
    description: 'console.error and [ERROR] tags',
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
  const listRef = useRef<FlatList<LogEntry> | null>(null);

  useEffect(() => {
    // Subscribe to log updates
    const listener = (newLogs: LogEntry[]) => {
      setLogs(newLogs);
    };

    logListeners.push(listener);

    return () => {
      logListeners = logListeners.filter(l => l !== listener);
    };
  }, []);

  const filteredLogs = useMemo(() => {
    switch (filter) {
      case 'yolo':
        return logs.filter((log) => {
          const messageUpper = log.message.toUpperCase();
          return (
            messageUpper.includes('[INFERENCE-DEBUG]')
            || messageUpper.includes('[PRIORITY-DEBUG]')
          );
        });
      case 'convlstm':
        return logs.filter((log) => log.message.toUpperCase().includes('[CONVLSTM-TRACE]'));
      case 'audio':
        return logs.filter((log) => {
          const messageUpper = log.message.toUpperCase();
          return (
            messageUpper.includes('[AUDIO-DEBUG]')
            || messageUpper.includes('[AUDIO-TRACE]')
          );
        });
      case 'errors':
        return logs.filter((log) => {
          const messageUpper = log.message.toUpperCase();
          return log.level === 'error' || messageUpper.includes('[ERROR]');
        });
      case 'all':
      default:
        return logs;
    }
  }, [filter, logs]);

  const selectedFilterOption = useMemo(
    () => LOG_FILTER_OPTIONS.find((option) => option.key === filter) ?? LOG_FILTER_OPTIONS[0],
    [filter],
  );

  useEffect(() => {
    if (!autoScroll || filteredLogs.length === 0) {
      return;
    }

    const timer = setTimeout(() => {
      listRef.current?.scrollToEnd({ animated: true });
    }, 80);

    return () => clearTimeout(timer);
  }, [autoScroll, filteredLogs.length]);

  const clearLogs = () => {
    logStorage = [];
    logIdCounter = 0;
    setLogs([]);
    logListeners.forEach((listener) => listener([]));
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
      case 'debug': return '#9dbb5b';
      default: return '#cccccc';
    }
  };

  const renderLogItem: ListRenderItem<LogEntry> = useCallback(({ item: log }) => (
    <View style={styles.logEntry}>
      <Text style={styles.logTime}>{formatTime(log.timestamp)}</Text>
      <Text style={[styles.logLevel, { color: getLevelColor(log.level) }]}>
        [{log.level.toUpperCase()}]
      </Text>
      <Text style={styles.logMessage}>{log.message}</Text>
    </View>
  ), []);

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

      {/* Filter Dropdown Header */}
      <View style={styles.filterContainer}>
        <Text style={styles.filterLabel}>Category</Text>
        <View style={styles.filterRow}>
          <TouchableOpacity
            style={styles.dropdownButton}
            onPress={() => setIsFilterModalVisible(true)}
            activeOpacity={0.85}
          >
            <View style={styles.dropdownTextWrap}>
              <Text style={styles.dropdownTitle}>{selectedFilterOption.label}</Text>
              <Text style={styles.dropdownDescription}>{selectedFilterOption.description}</Text>
            </View>
            <Text style={styles.dropdownCaret}>▼</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.clearHeaderButton}
            onPress={clearLogs}
            activeOpacity={0.85}
          >
            <Text style={styles.clearHeaderButtonText}>Clear</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Category Modal */}
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
            <Text style={styles.modalTitle}>Filter Logs</Text>
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
                <Text style={styles.modalOptionTitle}>{option.label}</Text>
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
      </View>

      {/* Logs Display */}
      <FlatList
        ref={listRef}
        data={filteredLogs}
        renderItem={renderLogItem}
        keyExtractor={(item) => String(item.id)}
        style={styles.logsContainer}
        contentContainerStyle={styles.logsContent}
        removeClippedSubviews
        initialNumToRender={30}
        maxToRenderPerBatch={60}
        windowSize={8}
        ListEmptyComponent={(
          <Text style={styles.emptyText}>No logs yet. Start using the app to see debug output.</Text>
        )}
      />

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
    paddingBottom: 6,
  },
  filterLabel: {
    color: '#90997b',
    fontSize: 12,
    marginBottom: 6,
    letterSpacing: 0.8,
  },
  filterRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
    gap: 8,
  },
  dropdownButton: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#36412a',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dropdownTextWrap: {
    flex: 1,
    marginRight: 8,
  },
  dropdownTitle: {
    color: '#f2f2f2',
    fontSize: 14,
    fontWeight: '600',
  },
  dropdownDescription: {
    color: '#8f9f75',
    fontSize: 11,
    marginTop: 2,
  },
  dropdownCaret: {
    color: '#9dbb5b',
    fontSize: 12,
  },
  clearHeaderButton: {
    minWidth: 76,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#3f2f2f',
    borderColor: '#6b4f4f',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
  },
  clearHeaderButtonText: {
    color: '#f4d6d6',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.65)',
  },
  modalCard: {
    backgroundColor: '#0f110b',
    borderWidth: 1,
    borderColor: '#3f4e2c',
    borderRadius: 12,
    padding: 12,
  },
  modalTitle: {
    color: '#f2f2f2',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 10,
  },
  modalOption: {
    backgroundColor: '#171a12',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2f3922',
    paddingVertical: 10,
    paddingHorizontal: 10,
    marginBottom: 8,
  },
  modalOptionActive: {
    borderColor: '#8ea95a',
    backgroundColor: '#2f3a22',
  },
  modalOptionTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  modalOptionDescription: {
    color: '#a0b584',
    fontSize: 11,
    marginTop: 2,
  },
  actionContainer: {
    flexDirection: 'row',
    paddingHorizontal: 10,
    paddingBottom: 8,
  },
  actionButton: {
    flex: 1,
    paddingVertical: 10,
    backgroundColor: '#1e2417',
    borderWidth: 1,
    borderColor: '#334024',
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonText: {
    color: '#d8e3c4',
    fontSize: 13,
    fontWeight: '600',
  },
  logsContainer: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  logsContent: {
    padding: 10,
    flexGrow: 1,
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
