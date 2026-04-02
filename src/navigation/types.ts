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
  Wandering: undefined;
  Destination: {
    origin?: { latitude: number; longitude: number };
    destination?: { latitude: number; longitude: number };
    destinationLabel?: string;
    routeSteps?: RouteStep[];
    totalDistanceMeters?: number;
    totalDurationSeconds?: number;
  } | undefined;
  Camera: undefined;
  Logs: undefined;
};

// Extend the navigation types
declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}
