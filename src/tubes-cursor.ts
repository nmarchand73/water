/**
 * TubesCursor overlay (threejs-components) — screen-space neon tubes following the pointer.
 * Renders on a separate canvas above the WebGPU pool; input stays on the main canvas.
 */
import TubesCursor from 'threejs-components/build/cursors/tubes1.min.js';

export type TubesCursorApp = ReturnType<typeof TubesCursor>;

/** Classic neon defaults (when “Match scene” is off). */
export const TUBES_DEFAULT_TUBE_COLORS = ['#f967fb', '#53bc28', '#6958d5'] as const;
export const TUBES_DEFAULT_LIGHT_COLORS = ['#83f36e', '#fe8a2e', '#ff008a', '#60aed5'] as const;

/** BC host from threejs-components — must call after syncing canvas pixels/CSS size. */
export function resizeTubesCursorHost(app: TubesCursorApp | null): void {
  const three = app?.three as { resize?: () => void } | undefined;
  three?.resize?.();
}

let app: TubesCursorApp | null = null;

/**
 * Create the Three.js tubes cursor on the overlay canvas (CodePen-style options).
 */
export function initTubesCursor(overlay: HTMLCanvasElement): TubesCursorApp {
  app = TubesCursor(overlay, {
    tubes: {
      colors: [...TUBES_DEFAULT_TUBE_COLORS],
      lights: {
        intensity: 250,
        colors: [...TUBES_DEFAULT_LIGHT_COLORS],
      },
    },
  });
  return app;
}

/**
 * Match overlay backing store and CSS size to the main WebGPU canvas.
 */
export function syncTubesCanvasSize(
  overlay: HTMLCanvasElement,
  mainCanvas: HTMLCanvasElement
): void {
  overlay.width = mainCanvas.width;
  overlay.height = mainCanvas.height;
  overlay.style.width = mainCanvas.style.width;
  overlay.style.height = mainCanvas.style.height;
}

function clonePointerEvent(e: PointerEvent): PointerEvent {
  return new PointerEvent(e.type, {
    bubbles: true,
    cancelable: true,
    composed: true,
    pointerId: e.pointerId,
    pointerType: e.pointerType,
    clientX: e.clientX,
    clientY: e.clientY,
    button: e.button,
    buttons: e.buttons,
    ctrlKey: e.ctrlKey,
    shiftKey: e.shiftKey,
    altKey: e.altKey,
    metaKey: e.metaKey,
    pressure: e.pressure,
    tiltX: e.tiltX,
    tiltY: e.tiltY,
    twist: e.twist,
    width: e.width,
    height: e.height,
    isPrimary: e.isPrimary,
    view: e.view ?? undefined,
    detail: e.detail,
  });
}

/**
 * TubesCursor attaches listeners to the overlay canvas. With `pointer-events: none`, events never
 * hit that canvas (they fall through to the WebGPU canvas), so the effect gets no input.
 * Enable hits on the overlay and clone pointer/wheel/contextmenu to the scene canvas so pool
 * interaction stays correct (same client coordinates; both canvases share layout).
 *
 * Register this **after** {@link initTubesCursor} so library listeners run first, then forwarding.
 */
export function wireOverlayPointerPassthrough(
  overlay: HTMLCanvasElement,
  sceneCanvas: HTMLCanvasElement
): void {
  overlay.style.pointerEvents = 'auto';

  const forwardPointer = (e: PointerEvent) => {
    sceneCanvas.dispatchEvent(clonePointerEvent(e));
  };

  overlay.addEventListener('pointerdown', forwardPointer);
  overlay.addEventListener('pointermove', forwardPointer);
  overlay.addEventListener('pointerup', forwardPointer);
  overlay.addEventListener('pointercancel', forwardPointer);

  overlay.addEventListener('wheel', (e) => {
    sceneCanvas.dispatchEvent(
      new WheelEvent('wheel', {
        bubbles: true,
        cancelable: true,
        clientX: e.clientX,
        clientY: e.clientY,
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        deltaZ: e.deltaZ,
        deltaMode: e.deltaMode,
        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,
      })
    );
  });

  overlay.addEventListener('contextmenu', (e) => {
    sceneCanvas.dispatchEvent(
      new MouseEvent('contextmenu', {
        bubbles: true,
        cancelable: true,
        clientX: e.clientX,
        clientY: e.clientY,
        button: e.button,
        buttons: e.buttons,
        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,
      })
    );
  });
}

const WATER_SURFACE_Y = 0;
const UNDERWATER_BLEND_M = 0.25;

/** RGB 0–1 */
type Rgb = readonly [number, number, number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mergeRgb(a: Rgb, b: Rgb, t: number): [number, number, number] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function toHex([r, g, b]: [number, number, number]): string {
  const c = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
  return `#${[c(r), c(g), c(b)].map((x) => x.toString(16).padStart(2, '0')).join('')}`;
}

/** Caustic-adjacent bases (cyan / aqua / violet) — reads as “on the water” not generic UI neon. */
const SCENE_TUBE_RGB: Rgb[] = [
  [0.42, 0.88, 0.96],
  [0.52, 0.76, 1.0],
  [0.58, 0.55, 0.98],
];

const SCENE_LIGHT_RGB: Rgb[] = [
  [0.52, 0.94, 0.74],
  [1.0, 0.88, 0.48],
  [0.98, 0.42, 0.62],
  [0.52, 0.82, 0.96],
];

const WARM_CAUSTIC: Rgb = [1.0, 0.93, 0.58];
const COOL_DEEP: Rgb = [0.28, 0.72, 0.88];

/**
 * Drive tube + light hex colors from scene sun direction and caustics slider (matches pool lighting mood).
 */
export function computeSceneIntegratedPalette(
  lightDir: { x: number; y: number; z: number },
  causticsIntensity: number
): { tubeColors: string[]; lightColors: string[] } {
  const le = Math.max(-1, Math.min(1, lightDir.y));
  const horizontalSun = Math.sqrt(Math.max(0, 1 - le * le));
  const warmT = horizontalSun * (0.22 + 0.58 * Math.max(0, Math.min(1, causticsIntensity)));
  const az = Math.atan2(lightDir.x, lightDir.z);
  const hueWalk = Math.cos(az) * 0.045;

  const tubeColors = SCENE_TUBE_RGB.map(([r, g, b], i) => {
    const shifted: [number, number, number] = [
      Math.min(1, r + hueWalk * (1 - i * 0.15)),
      g,
      Math.min(1, b - hueWalk * 0.35),
    ];
    let rgb = mergeRgb(shifted, WARM_CAUSTIC, warmT * 0.42);
    rgb = mergeRgb(rgb, COOL_DEEP, (1 - causticsIntensity) * 0.12);
    return toHex(rgb);
  });

  const lightColors = SCENE_LIGHT_RGB.map(([r, g, b], i) => {
    const rgb = mergeRgb([r, g, b], WARM_CAUSTIC, warmT * (0.38 + i * 0.04));
    return toHex(rgb);
  });

  return { tubeColors, lightColors };
}

let lastPaletteKey = '';
let wasSceneIntegration = false;

export type TubesScenePaletteOpts = {
  app: TubesCursorApp | null;
  enabled: boolean;
  sceneIntegration: boolean;
  lightDir: { x: number; y: number; z: number };
  causticsIntensity: number;
};

/**
 * Updates TubesCursor colors from sun + caustics. Cheap: skips when inputs unchanged.
 */
export function syncTubesCursorScenePalette(opts: TubesScenePaletteOpts): void {
  const tubesApi = opts.app?.tubes as
    | { setColors?: (c: string[]) => void; setLightsColors?: (c: string[]) => void }
    | undefined;
  if (!tubesApi?.setColors || !tubesApi.setLightsColors) return;

  if (!opts.enabled || !opts.sceneIntegration) {
    if (wasSceneIntegration) {
      tubesApi.setColors([...TUBES_DEFAULT_TUBE_COLORS]);
      tubesApi.setLightsColors([...TUBES_DEFAULT_LIGHT_COLORS]);
      lastPaletteKey = '';
    }
    wasSceneIntegration = false;
    return;
  }

  wasSceneIntegration = true;
  const key = `${opts.lightDir.x.toFixed(3)}_${opts.lightDir.y.toFixed(3)}_${opts.lightDir.z.toFixed(3)}_${opts.causticsIntensity.toFixed(2)}`;
  if (key === lastPaletteKey) return;
  lastPaletteKey = key;

  const { tubeColors, lightColors } = computeSceneIntegratedPalette(opts.lightDir, opts.causticsIntensity);
  tubesApi.setColors(tubeColors);
  tubesApi.setLightsColors(lightColors);
}

export type TubesOverlayOptions = {
  /** Master on/off (Settings → Cursor tubes). */
  enabled: boolean;
  /**
   * True when primary pointer is down (left click / touch) and the aim hits the pool
   * (add-ripples drag or press-and-hold over water). Caller gates hover-only.
   */
  waterContact: boolean;
  /** Camera eye world Y; water surface is y = 0. */
  eyeY: number;
  /** Opacity when fully below the surface (0 = hidden). */
  underwaterMinOpacity: number;
  /**
   * 0 = view ray nearly perpendicular to water (looking straight down at cursor); 1 = grazing / horizon.
   * Softer trail when grazing so the overlay feels glued to the surface, not a HUD.
   */
  grazingFactor: number;
  /** Small CSS hue-rotate (deg) from sun azimuth so the trail drifts with scene light. */
  lightHueRotateDeg: number;
  /**
   * Multiplier for opacity when the pointer is only over the pool (hover) vs actively drawing ripples.
   * Keeps hover subtler than click-drag.
   */
  pointerPresenceScale: number;
};

/**
 * Fade / dim the overlay when the camera is under the water surface (heuristic).
 */
export function updateTubesOverlayStyle(
  overlay: HTMLCanvasElement,
  opts: TubesOverlayOptions
): void {
  if (!opts.enabled) {
    overlay.style.opacity = '0';
    overlay.style.filter = 'none';
    return;
  }

  if (!opts.waterContact) {
    overlay.style.opacity = '0';
    overlay.style.filter = 'none';
    return;
  }

  let opacity = 1;
  if (opts.eyeY < WATER_SURFACE_Y) {
    const depth = WATER_SURFACE_Y - opts.eyeY;
    const u = Math.min(1, depth / UNDERWATER_BLEND_M);
    opacity = 1 + (opts.underwaterMinOpacity - 1) * u;
  }

  const grazing = Math.max(0, Math.min(1, opts.grazingFactor));
  opacity *= 1 - grazing * 0.28;
  opacity *= Math.max(0, Math.min(1, opts.pointerPresenceScale));

  overlay.style.opacity = String(Math.max(0, Math.min(1, opacity)));

  if (opts.eyeY < WATER_SURFACE_Y) {
    overlay.style.filter = 'saturate(0.65)';
  } else {
    const g = grazing;
    const sat = 0.88 + (1 - g) * 0.12;
    const hue = opts.lightHueRotateDeg;
    overlay.style.filter = `saturate(${sat.toFixed(3)}) hue-rotate(${hue.toFixed(1)}deg)`;
  }
}
