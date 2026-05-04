/** World-space half width/depth of the pool on X and Z (default). */
export const DEFAULT_POOL_HALF_EXTENT = 2.0;

/** Depth from water surface (y=0) to floor (default), in world units. */
export const DEFAULT_POOL_DEPTH = 1.0;

/** Ray/AABB upper Y for rim-style effects (default matches original demo). */
export const DEFAULT_POOL_RIM_MAX_Y = 2.0;

/** Default Beer-Lambert absorption (aligns with `liner-presets` classic light blue when presets are used). */
export const DEFAULT_WATER_ABSORPTION = 0.9;

/** Default interactive sphere radius. */
export const DEFAULT_BALL_RADIUS = 0.25;

/** UFO mesh / physics / water proxy scale vs the shared "ball radius" slider (33% larger). */
export const UFO_RADIUS_SCALE = 1.33;

/** Max "Wave response" in the GUI - explicit heightfield step goes unstable above ~2 (NaNs / device lost). */
export const MAX_WAVE_RESPONSE = 2.0;

/** @deprecated use DEFAULT_POOL_HALF_EXTENT */
export const POOL_HALF_EXTENT = DEFAULT_POOL_HALF_EXTENT;
