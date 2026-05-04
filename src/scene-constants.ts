/** World-space half width/depth of the pool on X and Z (default). */
export const DEFAULT_POOL_HALF_EXTENT = 2.0;

/**
 * Must match `REF_POOL_HALF_EXTENT` in `shaders/common/functions.wgsl` — scales simulated height (R
 * channel) to world meters via `simHeight * (poolHalfExtent / REF_POOL_HALF_EXTENT)`.
 */
export const REF_POOL_HALF_EXTENT = 2.0;

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

/**
 * Max "Wave response" in the GUI. Keep modest — explicit heightfield steps amplify quickly once the
 * Laplacian clamp allows larger neighborhoods to disagree (see MAX_WAVE_CURVATURE_STEP in WGSL).
 */
export const MAX_WAVE_RESPONSE = 1.08;

/** Simulation substeps per frame (matches original “two passes” when set to 2). */
export const WAVE_SIM_SUBSTEPS = 2;

/** Prior behavior: two full wave updates per frame — used to scale per-substep impulse. */
export const WAVE_LEGACY_STEPS_PER_FRAME = 2;

/** @deprecated use DEFAULT_POOL_HALF_EXTENT */
export const POOL_HALF_EXTENT = DEFAULT_POOL_HALF_EXTENT;
