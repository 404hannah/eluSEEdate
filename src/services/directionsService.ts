/**
 * OSRM Walking Directions Service (Open Source Routing Machine)
 *
 * Fetches step-by-step walking directions via the public OSRM demo server.
 * Free, no API key required.
 *
 * Each step contains start/end locations and a maneuver (turn instruction).
 * The returned RouteStep array is the raw material for building the Intent Feature
 * at runtime: maneuvers are one-hot encoded and combined with lat/lng + live GPS
 * to form the Intent Embedding for the ConvLSTM model.
 *
 * OSRM API: https://project-osrm.org/docs/v5.24.0/api/#route-service
 */

const OSRM_BASE = 'https://router.project-osrm.org';

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

/** Lat/lng coordinate pair. */
export type LatLng = {
  lat: number;
  lng: number;
};

/**
 * Walking maneuver values relevant to the study.
 * OSRM uses a type + modifier system; we map those to these 7 canonical classes.
 */
export type WalkingManeuver =
  | 'turn-left'
  | 'turn-slight-left'
  | 'turn-sharp-left'
  | 'turn-right'
  | 'turn-slight-right'
  | 'turn-sharp-right'
  | 'straight';

/** All known maneuvers in fixed order – used for one-hot encoding. */
export const MANEUVER_CLASSES: WalkingManeuver[] = [
  'turn-left',
  'turn-slight-left',
  'turn-sharp-left',
  'turn-right',
  'turn-slight-right',
  'turn-sharp-right',
  'straight',
];

/** A single segment of the walking route. */
export type RouteStep = {
  /** Index of this step in the route (0-based). */
  index: number;
  /** Where this segment begins. */
  startLocation: LatLng;
  /** Where this segment ends. */
  endLocation: LatLng;
  /** Human-readable instruction. */
  instruction: string;
  /** Distance in metres for this segment. */
  distanceMeters: number;
  /** Duration in seconds for this segment. */
  durationSeconds: number;
  /** The maneuver at the *end* of this segment. */
  maneuver: WalkingManeuver;
  /** One-hot encoded maneuver (length = MANEUVER_CLASSES.length = 7). */
  maneuverOneHot: number[];
};

/** Full result returned by fetchWalkingDirections. */
export type DirectionsResult = {
  /** Ordered array of route steps. */
  steps: RouteStep[];
  /** Total distance of the route in metres. */
  totalDistanceMeters: number;
  /** Total estimated walking duration in seconds. */
  totalDurationSeconds: number;
  /** Encoded polyline for the entire route (optional use). */
  overviewPolyline: string;
};

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

/**
 * Map an OSRM maneuver (type + modifier) to one of the 7 canonical classes.
 *
 * OSRM step format:
 *   maneuver.type     = "turn" | "new name" | "depart" | "arrive" | "continue" | ...
 *   maneuver.modifier = "left" | "slight left" | "sharp left" | "right" | "slight right" | "sharp right" | "straight" | "uturn" | ...
 */
function normalizeOsrmManeuver(type?: string, modifier?: string): WalkingManeuver {
  if (!modifier) return 'straight';

  const mod = modifier.toLowerCase();

  // Direct matches
  if (mod === 'sharp left') return 'turn-sharp-left';
  if (mod === 'slight left') return 'turn-slight-left';
  if (mod === 'left') return 'turn-left';
  if (mod === 'sharp right') return 'turn-sharp-right';
  if (mod === 'slight right') return 'turn-slight-right';
  if (mod === 'right') return 'turn-right';
  if (mod === 'straight') return 'straight';

  // Fallback heuristic
  if (mod.includes('left')) return 'turn-left';
  if (mod.includes('right')) return 'turn-right';
  return 'straight';
}

/** Create a one-hot vector for the given maneuver. */
function oneHotManeuver(m: WalkingManeuver): number[] {
  return MANEUVER_CLASSES.map((cls) => (cls === m ? 1 : 0));
}

/**
 * Build a short human-readable instruction from OSRM step data.
 */
function buildInstruction(step: any): string {
  const type: string = step.maneuver?.type ?? '';
  const modifier: string = step.maneuver?.modifier ?? '';
  const name: string = step.name ?? '';

  if (type === 'depart') return name ? `Head towards ${name}` : 'Depart';
  if (type === 'arrive') return 'Arrive at destination';

  const directionWord = modifier || 'straight';
  return name
    ? `${capitalize(directionWord)} onto ${name}`
    : `Continue ${directionWord}`;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ----------------------------------------------------------------
// Main function
// ----------------------------------------------------------------

/**
 * Fetch walking directions between two points using the OSRM public server.
 *
 * @param origin  { latitude, longitude }
 * @param destination  { latitude, longitude }
 * @returns Parsed DirectionsResult with steps, distances, maneuvers, and one-hot vectors.
 * @throws If OSRM returns an error or the network request fails.
 */
export async function fetchWalkingDirections(
  origin: { latitude: number; longitude: number },
  destination: { latitude: number; longitude: number },
): Promise<DirectionsResult> {
  // OSRM expects coordinates as lng,lat (note: longitude first)
  const coords = `${origin.longitude},${origin.latitude};${destination.longitude},${destination.latitude}`;
  const url =
    `${OSRM_BASE}/route/v1/foot/${coords}` +
    `?steps=true&overview=full&geometries=polyline&annotations=false`;

  const response = await fetch(url);
  const json = await response.json();

  if (json.code !== 'Ok') {
    throw new Error(`OSRM error: ${json.code} – ${json.message ?? 'unknown'}`);
  }

  const route = json.routes[0];
  const leg = route.legs[0]; // Single origin → destination = one leg

  const steps: RouteStep[] = leg.steps.map((step: any, idx: number) => {
    const maneuver = normalizeOsrmManeuver(
      step.maneuver?.type,
      step.maneuver?.modifier,
    );

    // OSRM provides maneuver.location as [lng, lat]
    const startLoc = step.maneuver?.location ?? [0, 0];
    // For end location, use the next step's maneuver location, or intersections
    const intersections = step.intersections ?? [];
    const lastIntersection = intersections[intersections.length - 1];
    const endLoc = lastIntersection?.location ?? startLoc;

    return {
      index: idx,
      startLocation: { lat: startLoc[1], lng: startLoc[0] } as LatLng,
      endLocation: { lat: endLoc[1], lng: endLoc[0] } as LatLng,
      instruction: buildInstruction(step),
      distanceMeters: step.distance ?? 0,
      durationSeconds: step.duration ?? 0,
      maneuver,
      maneuverOneHot: oneHotManeuver(maneuver),
    };
  });

  return {
    steps,
    totalDistanceMeters: route.distance ?? 0,
    totalDurationSeconds: route.duration ?? 0,
    overviewPolyline: route.geometry ?? '',
  };
}
