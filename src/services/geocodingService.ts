/**
 * Nominatim Geocoding Service (OpenStreetMap)
 *
 * Free geocoding with no API key required.
 * Converts place names → coordinates (forward) and coordinates → addresses (reverse).
 *
 * Usage policy: https://operations.osmfoundation.org/policies/nominatim/
 *   - Max 1 request per second (we only fire on user confirmation, so this is fine)
 *   - Provide a meaningful User-Agent
 */

const NOMINATIM_BASE = 'https://nominatim.openstreetmap.org';
const USER_AGENT = 'EluSEEdate-MobileApp/1.0 (thesis project)';

export type GeocodingResult = {
  latitude: number;
  longitude: number;
  displayName: string;
};

/**
 * Forward geocode: place name → coordinates + display name.
 * Returns the top result or null if nothing found.
 */
export async function geocodeForward(placeName: string): Promise<GeocodingResult | null> {
  const url =
    `${NOMINATIM_BASE}/search?` +
    `q=${encodeURIComponent(placeName)}` +
    `&format=json&limit=1&addressdetails=1`;

  const response = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Nominatim forward error: ${response.status}`);
  }

  const results = await response.json();
  if (!results.length) return null;

  const top = results[0];
  return {
    latitude: parseFloat(top.lat),
    longitude: parseFloat(top.lon),
    displayName: top.display_name as string,
  };
}

/**
 * Reverse geocode: coordinates → display name.
 * Returns a friendly address string or a lat/lng fallback.
 */
export async function geocodeReverse(
  latitude: number,
  longitude: number,
): Promise<string> {
  const url =
    `${NOMINATIM_BASE}/reverse?` +
    `lat=${latitude}&lon=${longitude}` +
    `&format=json&addressdetails=1`;

  const response = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
  });

  if (!response.ok) {
    return `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
  }

  const result = await response.json();
  return result.display_name ?? `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
}
