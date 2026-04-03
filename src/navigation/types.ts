/**
 * Navigation Type Definitions
 * 
 * Type-safe navigation with React Navigation
 */

import { RouteStep } from '../services/directionsService';

export type RootStackParamList = {
  MainMenu: undefined;
  Choice: undefined;
  Wayfinding: undefined;
  ActiveCamera: {
    mode: 'wandering' | 'destination';
    origin?: { latitude: number; longitude: number };
    destination?: { latitude: number; longitude: number };
    destinationLabel?: string;
    routeSteps?: RouteStep[];
    totalDistanceMeters?: number;
    totalDurationSeconds?: number;
  };
  Logs: undefined;
};

// Extend the navigation types
declare global {
  namespace ReactNavigation {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface RootParamList extends RootStackParamList {}
  }
}
