import { Vector } from './lightgl';

/** Relative ρ_object / ρ_water (water = 1). Hollow / air-filled shells sit near the low end. */
export const RELATIVE_DENSITY_MIN = 0.05;
export const RELATIVE_DENSITY_MAX = 2.0;

/** Typical inflatable ball: thin shell + air inside (average ρ vs water). */
export const DEFAULT_AIR_FILLED_SPHERE_DENSITY = 0.13;
/** Large UFO mesh: slightly lighter default than the small sphere. */
export const DEFAULT_AIR_FILLED_UFO_DENSITY = 0.12;

/** Interior preset: solid rubber / plastic (near-neutral to slightly buoyant). */
export const INTERIOR_SOLID_SPHERE_DENSITY = 0.92;
export const INTERIOR_SOLID_UFO_DENSITY = 0.86;
/** Interior preset: heavy (ρ &gt; water, sinks in the pool). */
export const INTERIOR_DENSE_SPHERE_DENSITY = 1.22;
export const INTERIOR_DENSE_UFO_DENSITY = 1.16;

/**
 * Water surface is the plane y = 0; the pool interior is y &lt; 0.
 * Returns the fraction of sphere volume in y &lt; 0 (spherical cap), for buoyancy and drag.
 */
export function submergedVolumeFraction(centerY: number, radius: number): number {
  return submergedVolumeFractionBelowPlane(centerY, radius, 0);
}

/**
 * Same spherical-cap fraction below a horizontal plane `planeY` (local water surface).
 * Legacy flat surface uses `planeY = 0`.
 */
export function submergedVolumeFractionBelowPlane(
  centerY: number,
  radius: number,
  planeY: number
): number {
  const R = Math.max(1e-6, radius);
  const bottom = centerY - R;
  const top = centerY + R;
  if (top <= planeY) {
    return 1;
  }
  if (bottom >= planeY) {
    return 0;
  }
  const h = planeY - bottom; // cap height from sphere bottom up to the plane
  const hClamped = Math.max(0, Math.min(2 * R, h));
  const vCap = (Math.PI * hClamped * hClamped * (3 * R - hClamped)) / 3;
  const vSphere = (4 / 3) * Math.PI * R * R * R;
  return vCap / vSphere;
}

/**
 * Sphere center Y such that the submerged volume fraction below horizontal plane `planeY`
 * equals `targetFraction` (Archimedes equilibrium uses targetFraction ≈ ρ_object/ρ_water for ρ ≤ 1).
 * Uses binary search; safe for light shells that should ride almost entirely above the surface.
 */
export function centerYForSubmergedFractionBelowPlane(
  planeY: number,
  radius: number,
  targetFraction: number
): number {
  const R = Math.max(1e-6, radius);
  const target = Math.max(0.001, Math.min(0.999, targetFraction));
  // Submerged fraction decreases as center.y increases; bracket [lowY, highY] with f(low) > target > f(high)
  let lo = planeY - 2.5 * R;
  let hi = planeY + 1.5 * R;
  for (let i = 0; i < 28; i++) {
    const mid = (lo + hi) * 0.5;
    const f = submergedVolumeFractionBelowPlane(mid, R, planeY);
    if (f > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) * 0.5;
}

/**
 * `rhoWater / rhoObject` when using relative density (water = 1). Lower ρ (hollow, foam,
 * thin shell + air) ⇒ larger factor ⇒ stronger upward reaction when submerged. When density
 * mode is off, use a slightly super-buoyant stand-in so the legacy look stays close to the old demo.
 */
export function buoyancyStrength(relativeObjectDensity: number, useDensity: boolean): number {
  if (useDensity) {
    const d = Math.max(
      RELATIVE_DENSITY_MIN,
      Math.min(RELATIVE_DENSITY_MAX, relativeObjectDensity)
    );
    return 1.0 / d;
  }
  return 1.1;
}

export type PoolBounds = { halfExtent: number; depth: number };

/**
 * Keeps a sphere center inside the pool floor and XZ walls; reflects velocity with damping.
 */
export function resolvePoolCollisions(
  center: Vector,
  velocity: Vector,
  r: number,
  pool: PoolBounds
): void {
  const h = pool.halfExtent - r;
  const floorY = r - pool.depth;
  const wallRestitution = 0.38;
  const floorRestitution = 0.7;

  if (center.x > h) {
    center.x = h;
    velocity.x = -Math.abs(velocity.x) * wallRestitution;
  } else if (center.x < -h) {
    center.x = -h;
    velocity.x = Math.abs(velocity.x) * wallRestitution;
  }

  if (center.z > h) {
    center.z = h;
    velocity.z = -Math.abs(velocity.z) * wallRestitution;
  } else if (center.z < -h) {
    center.z = -h;
    velocity.z = Math.abs(velocity.z) * wallRestitution;
  }

  if (center.y < floorY) {
    center.y = floorY;
    velocity.y = Math.abs(velocity.y) * floorRestitution;
  }
}
