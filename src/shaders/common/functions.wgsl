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
