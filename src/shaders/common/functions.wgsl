// Common utility functions used across multiple shaders

// Tile albedo on the pool floor: repeat pattern so each tile is half the linear size (2×2 tiles per former UV patch)
const FLOOR_TILE_UV_REPEAT : f32 = 2.0;

// Map world XZ in [-halfExtent, halfExtent] to simulation texture UV [0, 1]
fn poolXZToUv(xz : vec2f, halfExtent : f32) -> vec2f {
  let inv = 0.5 / halfExtent;
  return xz * inv + vec2f(0.5);
}

fn poolXZToFloorTileUv(xz : vec2f, halfExtent : f32) -> vec2f {
  return poolXZToUv(xz, halfExtent) * FLOOR_TILE_UV_REPEAT;
}

// Mirror UV into [0,1] for neighbor reads — approximate reflecting pool rim (Neumann / mirror height).
fn reflectPoolUv(c : vec2f) -> vec2f {
  var x = c.x;
  var y = c.y;
  if (x < 0.0) {
    x = -x;
  } else if (x > 1.0) {
    x = 2.0 - x;
  }
  if (y < 0.0) {
    y = -y;
  } else if (y > 1.0) {
    y = 2.0 - y;
  }
  return vec2f(x, y);
}

// Sim texture stores height in units tuned at this reference half-extent (see DEFAULT_POOL_HALF_EXTENT).
// Without scaling, larger pools look "waveless" because the same sim amplitude spans more XZ meters.
const REF_POOL_HALF_EXTENT : f32 = 2.0;

fn waterHeightWorld(simHeight : f32, poolHalfExtent : f32) -> f32 {
  return simHeight * (poolHalfExtent / REF_POOL_HALF_EXTENT);
}

// Ray-box intersection for pool walls
fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
  let tMin = (cubeMin - origin) / ray;
  let tMax = (cubeMax - origin) / ray;
  let t1 = min(tMin, tMax);
  let t2 = max(tMin, tMax);
  let tNear = max(max(t1.x, t1.y), t1.z);
  let tFar = min(min(t2.x, t2.y), t2.z);
  return vec2f(tNear, tFar);
}

// Beer-Lambert absorption in water: longer paths darken and shift blue-green (red absorbed faster).
// Coefficients are artistic (world units = meters at default pool scale).
fn waterExtinctionPerMeter() -> vec3f {
  return vec3f(0.24, 0.075, 0.045);
}

fn beerLambertTransmittance(distance : f32, strength : f32) -> vec3f {
  if (strength <= 0.0) {
    return vec3f(1.0);
  }
  let sigma = waterExtinctionPerMeter() * strength;
  return exp(-sigma * max(0.0, distance));
}

fn applyWaterAbsorption(rgb : vec3f, distance : f32, strength : f32) -> vec3f {
  return rgb * beerLambertTransmittance(distance, strength);
}

// --- Wave heightfield stability (R=height, G=vertical velocity in sim; B/A = normals in other passes) ---
// Fast sphere motion can inject huge per-frame deltas; the explicit wave step can diverge to Inf/NaN
// (black image, broken normals, or GPU timeout). Keep sim values bounded.
const MAX_WATER_SIM_HEIGHT : f32 = 0.95;
const MAX_WATER_SIM_VELOCITY : f32 = 3.85;

// Caps neighbor-average minus center per step — kills positive-feedback spikes (update.frag.wgsl).
const MAX_WAVE_CURVATURE_STEP : f32 = 0.068;

// Max height change from sphere displacement pass in one frame (sphere.frag.wgsl).
const MAX_SPHERE_DISPLACE_DELTA : f32 = 0.095;

fn clampWaterSimRg(info : vec4f) -> vec4f {
  return vec4f(
    clamp(info.r, -MAX_WATER_SIM_HEIGHT, MAX_WATER_SIM_HEIGHT),
    clamp(info.g, -MAX_WATER_SIM_VELOCITY, MAX_WATER_SIM_VELOCITY),
    info.b,
    info.a
  );
}
