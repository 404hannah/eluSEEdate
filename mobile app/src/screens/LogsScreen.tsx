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
} from 'react-native';
import { useNavigation } from '@react-navigation/native';

// Global log storage
interface LogEntry {
  id: number;
  timestamp: Date;
  level: 'log' | 'warn' | 'error' | 'info' | 'debug';
  message: string;
}

let logStorage: LogEntry[] = [];
let logIdCounter = 0;
let logListeners: ((logs: LogEntry[]) => void)[] = [];

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
  const [filter, setFilter] = useState<'all' | 'yolo' | 'convlstm' | 'errors'>('all');
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
        return logs.filter(log => 
          log.message.toLowerCase().includes('yolo') ||
          log.message.toLowerCase().includes('detection')
        );
      case 'convlstm':
        return logs.filter(log => 
          log.message.toLowerCase().includes('convlstm') ||
          log.message.toLowerCase().includes('prediction') ||
          log.message.toLowerCase().includes('intent')
        );
      case 'errors':
        return logs.filter(log => log.level === 'error' || log.level === 'warn');
      default:
        return logs;
    }
  };

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

      {/* Filter Buttons */}
      <View style={styles.filterContainer}>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'all' && styles.filterButtonActive]}
          onPress={() => setFilter('all')}
        >
          <Text style={styles.filterButtonText}>All ({logs.length})</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'yolo' && styles.filterButtonActive]}
          onPress={() => setFilter('yolo')}
        >
          <Text style={styles.filterButtonText}>YOLO</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'convlstm' && styles.filterButtonActive]}
          onPress={() => setFilter('convlstm')}
        >
          <Text style={styles.filterButtonText}>ConvLSTM</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'errors' && styles.filterButtonActive]}
          onPress={() => setFilter('errors')}
        >
          <Text style={styles.filterButtonText}>Errors</Text>
        </TouchableOpacity>
      </View>

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
    flexDirection: 'row',
    padding: 10,
    gap: 8,
  },
  filterButton: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#222',
    borderRadius: 8,
    alignItems: 'center',
  },
  filterButtonActive: {
    backgroundColor: '#0a7ea4',
  },
  filterButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
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
