/**
 * Typical vinyl liner "water color" families (industry-style categories, not trademarked SKUs).
 * Tints scale sampled tiles and underwater shading toward common pool water tones.
 */
export type LinerRgb = readonly [number, number, number];

export type LinerPresetDefinition = {
  readonly label: string;
  readonly underTint: LinerRgb;
  readonly aboveTint: LinerRgb;
  readonly tileTint: LinerRgb;
  readonly waterAbsorption: number;
};

/** Keys match GUI dropdown values */
export const LINER_PRESET_IDS = [
  'classic_light_blue',
  'medium_blue',
  'deep_blue',
  'aqua',
  'sand_lagoon',
  'gray_graphite',
  'black_midnight',
] as const;

export type LinerPresetId = (typeof LINER_PRESET_IDS)[number];

export const LINER_PRESETS: Record<LinerPresetId, LinerPresetDefinition> = {
  classic_light_blue: {
    label: 'Classic light blue',
    underTint: [0.4, 0.9, 1.0],
    aboveTint: [0.25, 1.0, 1.25],
    tileTint: [1.02, 1.06, 1.1],
    waterAbsorption: 0.9,
  },
  medium_blue: {
    label: 'Medium blue',
    underTint: [0.32, 0.78, 0.96],
    aboveTint: [0.22, 0.88, 1.08],
    tileTint: [0.96, 1.01, 1.06],
    waterAbsorption: 1.0,
  },
  deep_blue: {
    label: 'Deep blue',
    underTint: [0.22, 0.58, 0.88],
    aboveTint: [0.16, 0.68, 0.96],
    tileTint: [0.88, 0.94, 1.02],
    waterAbsorption: 1.2,
  },
  aqua: {
    label: 'Aqua / turquoise',
    underTint: [0.38, 0.94, 0.9],
    aboveTint: [0.32, 1.0, 0.94],
    tileTint: [0.92, 1.04, 1.0],
    waterAbsorption: 0.75,
  },
  sand_lagoon: {
    label: 'Sand / lagoon',
    underTint: [0.52, 0.82, 0.74],
    aboveTint: [0.44, 0.88, 0.78],
    tileTint: [1.06, 0.98, 0.88],
    waterAbsorption: 0.65,
  },
  gray_graphite: {
    label: 'Gray / graphite',
    underTint: [0.44, 0.48, 0.52],
    aboveTint: [0.38, 0.44, 0.48],
    tileTint: [0.76, 0.78, 0.82],
    waterAbsorption: 1.15,
  },
  black_midnight: {
    label: 'Black / midnight',
    underTint: [0.12, 0.14, 0.18],
    aboveTint: [0.1, 0.12, 0.15],
    tileTint: [0.38, 0.4, 0.44],
    waterAbsorption: 1.55,
  },
};

export const DEFAULT_LINER_PRESET_ID: LinerPresetId = 'classic_light_blue';
