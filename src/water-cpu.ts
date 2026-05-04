import { REF_POOL_HALF_EXTENT } from './scene-constants';

/** Matches WGSL `waterHeightWorld(simHeight, poolHalfExtent)`. */
export function waterHeightWorld(simHeight: number, poolHalfExtent: number): number {
  return simHeight * (poolHalfExtent / REF_POOL_HALF_EXTENT);
}

/** Last completed GPU readback of sim height (+ slopes) at the floater’s XZ — may lag 1–2 frames. */
export type WaveCpuSample = {
  valid: boolean;
  /** Heightfield value R at the sample texel (sim units). */
  simH: number;
  /** Local water surface Y in world units at sample position. */
  surfaceWorldY: number;
  dhdxWorld: number;
  dhdzWorld: number;
};

export const EMPTY_WAVE_CPU_SAMPLE: WaveCpuSample = {
  valid: false,
  simH: 0,
  surfaceWorldY: 0,
  dhdxWorld: 0,
  dhdzWorld: 0,
};
