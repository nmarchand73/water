/**
 * main.ts - WebGPU Water Simulation Entry Point
 *
 * This is the main entry point for the interactive water simulation demo.
 * It initializes WebGPU, loads resources, sets up event handlers, and runs
 * the main render loop.
 *
 * Features:
 * - Interactive water ripples (click/drag on water surface)
 * - Draggable sphere with physics (gravity, buoyancy from submerged volume, pool collisions)
 * - Orbit camera controls (drag on empty space)
 * - Dynamic lighting (hold L key to adjust light direction)
 * - Pause/resume simulation (spacebar)
 *
 * Controls:
 * - Click on water: Add ripple
 * - Drag on water: Add multiple ripples
 * - Drag on sphere: Move sphere
 * - Drag elsewhere: Rotate camera
 * - Mouse wheel: Zoom in/out
 * - G key: Toggle gravity/physics on sphere
 * - L key (hold): Adjust light direction with camera
 * - Spacebar: Pause/resume simulation
 */

import { mat4, vec3 } from 'wgpu-matrix';
import GUI from 'lil-gui';
import { Pool } from './pool';
import { Sphere } from './sphere';
import { Water } from './water';
import { Vector, Raytracer } from './lightgl';
import { Cubemap } from './cubemap';
import { InteractionMode } from './types';
import type { MatricesPair, Viewport } from './types';
import {
  DEFAULT_BALL_RADIUS,
  DEFAULT_POOL_DEPTH,
  DEFAULT_POOL_HALF_EXTENT,
  DEFAULT_POOL_RIM_MAX_Y,
  UFO_RADIUS_SCALE,
  MAX_WAVE_RESPONSE,
  WAVE_LEGACY_STEPS_PER_FRAME,
  WAVE_SIM_SUBSTEPS,
} from './scene-constants';
import {
  DEFAULT_LINER_PRESET_ID,
  LINER_PRESETS,
  type LinerPresetId,
} from './liner-presets';
import ufoObjUrl from './shapes/UFO_Saucer.obj?url';
import { fetchAndParseUfoObj } from './shapes/parse-ufo-obj';
import {
  buoyancyStrength,
  centerYForSubmergedFractionBelowPlane,
  DEFAULT_AIR_FILLED_SPHERE_DENSITY,
  DEFAULT_AIR_FILLED_UFO_DENSITY,
  INTERIOR_DENSE_SPHERE_DENSITY,
  INTERIOR_DENSE_UFO_DENSITY,
  INTERIOR_SOLID_SPHERE_DENSITY,
  INTERIOR_SOLID_UFO_DENSITY,
  RELATIVE_DENSITY_MAX,
  RELATIVE_DENSITY_MIN,
  resolvePoolCollisions,
  submergedVolumeFraction,
  submergedVolumeFractionBelowPlane,
} from './sphere-physics';
import {
  initTubesCursor,
  resizeTubesCursorHost,
  syncTubesCanvasSize,
  syncTubesCursorScenePalette,
  updateTubesOverlayStyle,
  wireOverlayPointerPassthrough,
  type TubesCursorApp,
} from './tubes-cursor';

/** GUI interior presets (built-in ρ vs water); Custom uses the slider. */
type InteriorPreset = 'Air-filled' | 'Solid' | 'Dense' | 'Custom';

/**
 * Main initialization function.
 * Sets up WebGPU, loads resources, and starts the render loop.
 */
async function init(): Promise<void> {
  // --- WebGPU Initialization ---

  const gpu = navigator.gpu;
  if (!gpu) {
    document.getElementById('loading')!.textContent = 'WebGPU not supported.';
    return;
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    document.getElementById('loading')!.textContent = 'No WebGPU adapter found.';
    return;
  }

  // Request float32-filterable feature if available (better precision for water simulation)
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('float32-filterable')) {
    requiredFeatures.push('float32-filterable');
  }

  const device = await adapter.requestDevice({ requiredFeatures });
  const viewArea = document.getElementById('view-area') as HTMLDivElement;
  const canvas = document.getElementById('scene-canvas') as HTMLCanvasElement;
  const tubesCanvas = document.getElementById('tubes-cursor') as HTMLCanvasElement;
  let tubesApp: TubesCursorApp | null = null;
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({ device, format, alphaMode: 'premultiplied' });

  // --- State Variables ---

  const help = document.getElementById('help')!;
  let prevTime = performance.now();

  // --- Texture Loading ---

  /**
   * Loads an image from URL and creates a WebGPU texture.
   * @param url - Path to the image file
   * @returns Promise resolving to the created GPUTexture
   */
  async function loadTexture(url: string): Promise<GPUTexture> {
    const res = await fetch(url);
    const blob = await res.blob();
    const source = await createImageBitmap(blob);

    const texture = device.createTexture({
      label: url,
      size: [source.width, source.height],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
      { source, flipY: true },
      { texture },
      { width: source.width, height: source.height }
    );

    return texture;
  }

  // Base URL for assets (handles Vite dev server and production build)
  const base = import.meta.env.BASE_URL as string;

  // Load tile texture for pool walls
  const tileTexture = await loadTexture(`${base}tiles.jpg`);
  const tileSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

  // Load skybox cubemap for reflections
  const cubemap = new Cubemap(device);
  const skyTexture = await cubemap.load({
    xpos: `${base}xpos.jpg`,
    xneg: `${base}xneg.jpg`,
    ypos: `${base}ypos.jpg`,
    yneg: `${base}yneg.jpg`,
    zpos: `${base}zpos.jpg`,
    zneg: `${base}zneg.jpg`,
  });
  const skySampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // --- Camera State ---

  /** Camera pitch angle in degrees */
  let angleX = -25;
  /** Camera yaw angle in degrees */
  let angleY = -200.5;
  /** Camera distance from center */
  let distance = 4;

  /** Target camera pitch (for damping) */
  let targetAngleX = angleX;
  /** Target camera yaw (for damping) */
  let targetAngleY = angleY;
  /** Target camera distance (for damping) */
  let targetDistance = distance;

  /**
   * Computes the current view and projection matrices based on camera angles.
   * @returns Object containing projectionMatrix and viewMatrix
   */
  function getMatrices(): MatricesPair {
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(Math.PI / 4, aspect, 0.01, 100);

    // Build view matrix: translate back, rotate, translate up
    const viewMatrix = mat4.identity();
    mat4.translate(viewMatrix, [0, 0, -distance], viewMatrix); // Camera distance
    mat4.rotateX(viewMatrix, (-angleX * Math.PI) / 180, viewMatrix); // Pitch
    mat4.rotateY(viewMatrix, (-angleY * Math.PI) / 180, viewMatrix); // Yaw
    mat4.translate(viewMatrix, [0, 0.5, 0], viewMatrix); // Look slightly above center

    return { projectionMatrix, viewMatrix };
  }

  // --- Uniform Buffers ---

  // Common uniforms: view-projection matrix (64 bytes) + eye position (12 bytes) + padding (4 bytes)
  const uniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Light direction (vec3 + padding = 16 bytes)
  const lightUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // SphereUniforms: center, radius, spinY, shapeKind, wavePitch, waveRoll (32 bytes - bindings.wgsl)
  const sphereUniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Shadow toggle flags (3 floats + padding = 16 bytes)
  const shadowUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Water rendering uniforms — see WaterUniforms in bindings.wgsl (32 bytes)
  const waterUniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Pool / scene scale + liner-style water tints (`SceneParams`, see bindings.wgsl)
  const sceneParamsBuffer = device.createBuffer({
    label: 'SceneParams',
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const defaultLiner = LINER_PRESETS[DEFAULT_LINER_PRESET_ID];

  const linerAppearance = {
    underTint: [...defaultLiner.underTint] as [number, number, number],
    tileTint: [...defaultLiner.tileTint] as [number, number, number],
    aboveTint: [...defaultLiner.aboveTint] as [number, number, number],
  };

  const sceneDims = {
    poolHalfExtent: DEFAULT_POOL_HALF_EXTENT,
    poolDepth: DEFAULT_POOL_DEPTH,
    poolRimMaxY: DEFAULT_POOL_RIM_MAX_Y,
    ballRadius: DEFAULT_BALL_RADIUS,
    waterAbsorption: defaultLiner.waterAbsorption,
  };

  function syncSceneParams(): void {
    const d = new Float32Array(16);
    d[0] = sceneDims.poolHalfExtent;
    d[1] = sceneDims.poolDepth;
    d[2] = sceneDims.poolRimMaxY;
    d[3] = sceneDims.waterAbsorption;
    d[4] = linerAppearance.underTint[0];
    d[5] = linerAppearance.underTint[1];
    d[6] = linerAppearance.underTint[2];
    d[7] = 1.0;
    d[8] = linerAppearance.tileTint[0];
    d[9] = linerAppearance.tileTint[1];
    d[10] = linerAppearance.tileTint[2];
    d[11] = 1.0;
    d[12] = linerAppearance.aboveTint[0];
    d[13] = linerAppearance.aboveTint[1];
    d[14] = linerAppearance.aboveTint[2];
    d[15] = 1.0;
    device.queue.writeBuffer(sceneParamsBuffer, 0, d);
  }
  syncSceneParams();

  // --- Lighting ---

  /** Current light direction (normalized) */
  let lightDir = new Vector(2.0, 2.0, -1.0).unit();

  /**
   * Updates the light direction uniform buffer.
   */
  function updateLight(): void {
    device.queue.writeBuffer(lightUniformBuffer, 0, new Float32Array([...lightDir.toArray(), 0]));
  }
  updateLight();

  // Initialize shadow flags (all enabled: rim=1, sphere=1, ao=1)
  device.queue.writeBuffer(shadowUniformBuffer, 0, new Float32Array([1.0, 1.0, 1.0, 0.0]));

  // --- Scene Objects ---

  // Create pool (walls and floor)
  const pool = new Pool(
    device,
    format,
    uniformBuffer,
    tileTexture,
    tileSampler,
    lightUniformBuffer,
    sphereUniformBuffer,
    shadowUniformBuffer,
    sceneParamsBuffer,
    sceneDims.poolHalfExtent
  );

  // Create interactive sphere
  const sphere = new Sphere(
    device,
    format,
    uniformBuffer,
    lightUniformBuffer,
    sphereUniformBuffer,
    sceneParamsBuffer
  );

  try {
    const ufoMesh = await fetchAndParseUfoObj(ufoObjUrl);
    sphere.setUfoMeshFromData(ufoMesh);
  } catch (e) {
    console.warn('Could not load shapes/UFO_Saucer.obj; using built-in saucer mesh.', e);
  }

  // Create water simulation (256x256 resolution)
  const water = new Water(
    device,
    256,
    256,
    uniformBuffer,
    lightUniformBuffer,
    sphereUniformBuffer,
    shadowUniformBuffer,
    waterUniformBuffer,
    tileTexture,
    tileSampler,
    skyTexture,
    skySampler,
    sceneParamsBuffer,
    sceneDims.poolHalfExtent
  );

  /** Wave simulation tuning (Settings, Wave simulation folder). */
  const waveSim = {
    sphereInject: 0.1,
    /** Neighbor-coupling gain (per substep; shader clamps curvature / velocity). */
    waveResponse: 0.48,
    damping: 0.9974,
  };
  waveSim.waveResponse = Math.min(waveSim.waveResponse, MAX_WAVE_RESPONSE);

  /**
   * Per-substep GPU uniforms: no frame-dt scaling (matches original stable demo); curvature is
   * clamped in the shader. Legacy 2×/frame impulse split across N substeps; damping split per substep.
   */
  function computeWaveStepUniforms(): {
    waveResponseStep: number;
    dampingStep: number;
  } {
    const waveResponseStep =
      waveSim.waveResponse * (WAVE_LEGACY_STEPS_PER_FRAME / WAVE_SIM_SUBSTEPS);
    const dampingStep = Math.pow(waveSim.damping, 1.0 / WAVE_SIM_SUBSTEPS);
    return { waveResponseStep, dampingStep };
  }

  const waveSimGui: {
    inject?: { updateDisplay: () => void };
    wave?: { updateDisplay: () => void };
    damp?: { updateDisplay: () => void };
  } = {};

  function applyWaveSimulationParams(): void {
    const { waveResponseStep, dampingStep } = computeWaveStepUniforms();
    water.setSimulationParams(waveSim.sphereInject, waveResponseStep, dampingStep);
    waveSimGui.inject?.updateDisplay();
    waveSimGui.wave?.updateDisplay();
    waveSimGui.damp?.updateDisplay();
  }

  // --- Sphere Physics State ---

  /** Current sphere center position */
  let center = new Vector(-0.4, -0.75, 0.2);
  /** Previous frame sphere position (for water displacement) */
  let oldCenter = center.clone();
  /** Current sphere velocity */
  let velocity = new Vector();
  /** Whether physics (gravity/buoyancy) is enabled */
  let useSpherePhysics = true;
  /** Whether simulation is paused */
  let paused = false;

  /** Pointer interaction mode — declared early so `writeSphereUniforms` can read it during init. */
  let mode: InteractionMode = InteractionMode.None;

  /** Accumulated Y rotation for UFO mesh (radians). */
  let ufoSpin = 0;
  /**
   * Passive Y spin for the plain sphere (radians). A uniform sphere is symmetric, so
   * `spinY = 0` makes it look "frozen" next to the spinning saucer; a slow spin makes
   * the triplanar grain and lighting read motion (wave pitch/roll still apply on top).
   */
  let sphereSpin = 0;

  /** Smoothed wave-follow tilt (radians), aligned with water surface normal via dhdx/dhdz readback. */
  let waveTiltPitch = 0;
  let waveTiltRoll = 0;

  // --- GUI ---
  const gui = new GUI({ title: 'Settings' });
  gui.close(); // Collapse by default

  const waterFolder = gui.addFolder('Water');
  const objectFolder = gui.addFolder('Object');
  const sceneFolder = gui.addFolder('Scene');
  const waveSimFolder = gui.addFolder('Wave simulation');

  const linerPresetLabels = (Object.keys(LINER_PRESETS) as LinerPresetId[]).map(
    (id) => LINER_PRESETS[id].label
  );
  const linerLabelToId = new Map(
    (Object.keys(LINER_PRESETS) as LinerPresetId[]).map((id) => [
      LINER_PRESETS[id].label,
      id,
    ])
  );

  const settings = {
    gravity: useSpherePhysics,
    followCamera: false,
    object: 'Sphere',
    useDensity: true,
    /** Air-filled / Solid / Dense use preset ρ (sphere vs UFO); Custom uses the slider. */
    interior: 'Solid' as InteriorPreset,
    density: INTERIOR_SOLID_SPHERE_DENSITY,
    causticsIntensity: 0.2,
    ior: 1.333,
    fresnelMin: 0.25,
    surfaceRoughness: 0.12,
    foamStrength: 0.55,
    linerPreset: defaultLiner.label,
    /** TubesCursor overlay (threejs-components); CC BY-NC-SA — see README. */
    cursorTubes: false,
    /** Opacity when the camera is fully underwater (below y = 0). */
    tubesUnderwaterMinOpacity: 0.18,
    /** Tube colors + CSS follow sun direction and caustics (pool-integrated look). */
    tubesMatchScene: true,
    /** Sample GPU wave height at the floater XZ and couple buoyancy + slope drift (1-frame lag). */
    waveBodyCoupling: true,
  };

  /**
   * Vertical spring toward local surface height. Buoyancy tracks fSub; keep moderate — high values
   * + 1-frame GPU lag drive vertical oscillation (especially air-filled).
   */
  const WAVE_RIDE_SPRING = 9.5;
  /** Horizontal push along wave slope (noise-sensitive — clamp slopes below). */
  const WAVE_SLOPE_ACCEL = 2.05;
  /** Ignore spike gradients from 3×3 readback (reduces jitter / “surfing” ripples). */
  const WAVE_MAX_SURFACE_SLOPE = 0.55;

  /** Cap hull tilt so extreme slopes don’t flip the mesh (~±30°). */
  const WAVE_MAX_BODY_TILT_RAD = 0.52;

  /** Extra gain so shallow swell still reads on camera (atan slope alone is tiny). */
  const WAVE_BODY_TILT_GAIN = 2.75;

  /**
   * Euler tilt toward normalized surface (-dhdx, 1, -dhdz): roll about Z from dhdx, pitch about X from dhdz.
   * Clamped for stability; smoothing happens in `writeSphereUniforms`.
   */
  function targetWaveTiltFromSlopes(dhdx: number, dhdz: number): { roll: number; pitch: number } {
    const sx = Math.max(
      -WAVE_MAX_SURFACE_SLOPE,
      Math.min(WAVE_MAX_SURFACE_SLOPE, dhdx)
    );
    const sz = Math.max(
      -WAVE_MAX_SURFACE_SLOPE,
      Math.min(WAVE_MAX_SURFACE_SLOPE, dhdz)
    );
    let roll = Math.atan2(sx, 1.0) * WAVE_BODY_TILT_GAIN;
    let pitch = Math.atan2(-sz, 1.0) * WAVE_BODY_TILT_GAIN;
    const slopeBoost = 1.0 + Math.min(1.8, Math.hypot(sx, sz) * 3.2);
    roll *= slopeBoost;
    pitch *= slopeBoost;
    return {
      roll: Math.max(-WAVE_MAX_BODY_TILT_RAD, Math.min(WAVE_MAX_BODY_TILT_RAD, roll)),
      pitch: Math.max(-WAVE_MAX_BODY_TILT_RAD, Math.min(WAVE_MAX_BODY_TILT_RAD, pitch)),
    };
  }

  const gravityController = objectFolder
    .add(settings, 'gravity')
    .name('Toggle Gravity')
    .onChange((v: boolean) => {
      useSpherePhysics = v;
      (document.activeElement as HTMLElement)?.blur();
    });

  objectFolder
    .add(settings, 'waveBodyCoupling')
    .name('Wave ↔ body coupling')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  objectFolder
    .add(settings, 'useDensity')
    .name('Use object density')
    .onChange(() => {
      if (settings.useDensity) {
        syncDensityFromInterior();
      }
      updateDensityVisibility();
      (document.activeElement as HTMLElement)?.blur();
    });

  const interiorController = objectFolder
    .add(settings, 'interior', ['Air-filled', 'Solid', 'Dense', 'Custom'])
    .name('Interior')
    .onChange(() => {
      syncDensityFromInterior();
      updateDensityVisibility();
      (document.activeElement as HTMLElement)?.blur();
    });

  const densitySlider = objectFolder
    .add(settings, 'density', RELATIVE_DENSITY_MIN, RELATIVE_DENSITY_MAX, 0.01)
    .name('Rel. density (custom)')
    .onChange(() => {
      settings.interior = 'Custom';
      interiorController.updateDisplay();
      (document.activeElement as HTMLElement)?.blur();
    });

  /** Preset relative density for current object shape (Sphere vs UFO scales differ slightly). */
  function interiorPresetDensityForObject(): number {
    const ufo = settings.object === 'UFO';
    switch (settings.interior as InteriorPreset) {
      case 'Air-filled':
        return ufo ? DEFAULT_AIR_FILLED_UFO_DENSITY : DEFAULT_AIR_FILLED_SPHERE_DENSITY;
      case 'Solid':
        return ufo ? INTERIOR_SOLID_UFO_DENSITY : INTERIOR_SOLID_SPHERE_DENSITY;
      case 'Dense':
        return ufo ? INTERIOR_DENSE_UFO_DENSITY : INTERIOR_DENSE_SPHERE_DENSITY;
      default:
        return settings.density;
    }
  }

  /** ρ_object/ρ_water for physics + water shading when density mode is on. */
  function effectiveObjectDensity(): number {
    if (!settings.useDensity) {
      return 1.0;
    }
    if ((settings.interior as InteriorPreset) === 'Custom') {
      return settings.density;
    }
    return interiorPresetDensityForObject();
  }

  function syncDensityFromInterior(): void {
    if ((settings.interior as InteriorPreset) !== 'Custom') {
      settings.density = interiorPresetDensityForObject();
    }
    densitySlider.updateDisplay();
  }

  /**
   * Interior row when density mode is on; custom slider only for Custom interior.
   */
  function updateDensityVisibility(): void {
    interiorController.show(settings.useDensity);
    const showCustom =
      settings.useDensity && settings.interior === 'Custom';
    densitySlider.show(showCustom);
  }

  syncDensityFromInterior();
  updateDensityVisibility();

  /** Visual + physics + water radius for the current object (UFO is scaled up vs sphere). */
  function effectiveObjectRadius(): number {
    return settings.object === 'UFO'
      ? sceneDims.ballRadius * UFO_RADIUS_SCALE
      : sceneDims.ballRadius;
  }

  /** Writes sphere/UFO GPU uniforms; advances spin when not paused. */
  function writeSphereUniforms(frameDt: number): void {
    if (frameDt > 0 && !paused) {
      if (settings.object === 'UFO') {
        ufoSpin += frameDt * 2.4;
      } else if (settings.object === 'Sphere') {
        // Slower than saucer: hull is round; still enough for grain + highlights to move.
        sphereSpin += frameDt * 0.72;
      }
    }

    // Snappier than buoy smoothing — hull tilt must catch visible swell.
    const tiltAlpha = 1.0 - Math.exp(-18.0 * Math.min(frameDt, 0.12));

    if (
      !paused &&
      objectIsSolid() &&
      settings.waveBodyCoupling &&
      mode !== InteractionMode.MoveSphere
    ) {
      const w = water.getWaveCpuSample();
      if (w.valid) {
        const { roll: targetRoll, pitch: targetPitch } = targetWaveTiltFromSlopes(
          w.dhdxWorld,
          w.dhdzWorld
        );
        const r = effectiveObjectRadius();
        const fSub = submergedVolumeFractionBelowPlane(
          center.y,
          r,
          w.surfaceWorldY
        );
        // Air-filled floaters are mostly above water → fSub alone killed tilt. Weight by
        // vertical distance to local surface so “riding the swell” still rocks the hull.
        const dy = Math.abs(center.y - w.surfaceWorldY);
        const surfaceProximity = Math.exp(-dy / Math.max(1e-6, r * 0.75));
        const tiltWeight = Math.max(
          0.38,
          Math.min(1, 0.18 + 0.82 * Math.max(fSub, surfaceProximity * 0.92))
        );
        waveTiltRoll += (targetRoll * tiltWeight - waveTiltRoll) * tiltAlpha;
        waveTiltPitch += (targetPitch * tiltWeight - waveTiltPitch) * tiltAlpha;
      } else {
        waveTiltRoll *= 1.0 - tiltAlpha * 0.55;
        waveTiltPitch *= 1.0 - tiltAlpha * 0.55;
      }
    } else {
      const relax = 1.0 - tiltAlpha * 0.88;
      waveTiltRoll *= relax;
      waveTiltPitch *= relax;
    }

    sphere.setMeshKind(settings.object === 'UFO' ? 'ufo' : 'sphere');
    const shapeKind = settings.object === 'UFO' ? 1 : 0;
    const spinY = settings.object === 'UFO' ? ufoSpin : sphereSpin;
    sphere.update(
      center.toArray(),
      effectiveObjectRadius(),
      spinY,
      shapeKind,
      waveTiltPitch,
      waveTiltRoll
    );
  }

  function objectIsSolid(): boolean {
    return settings.object === 'Sphere' || settings.object === 'UFO';
  }

  objectFolder
    .add(settings, 'object', ['Sphere', 'UFO', 'None'])
    .name('Object')
    .onChange((v: string) => {
      const isVisible = v === 'Sphere' || v === 'UFO';
      syncDensityFromInterior();
      // Update shadow flags when sphere visibility changes: rim=1, sphere=isVisible, ao=1
      device.queue.writeBuffer(
        shadowUniformBuffer,
        0,
        new Float32Array([1.0, isVisible ? 1.0 : 0.0, 1.0, 0.0])
      );
      const r = effectiveObjectRadius();
      center.x = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.x)
      );
      center.y = Math.max(
        r - sceneDims.poolDepth,
        Math.min(10, center.y)
      );
      center.z = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.z)
      );
      writeSphereUniforms(0);
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(settings, 'followCamera')
    .name('Light From Camera')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(settings, 'cursorTubes')
    .name('Cursor tubes')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(settings, 'tubesUnderwaterMinOpacity', 0, 1, 0.02)
    .name('Tubes underwater min α')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(settings, 'tubesMatchScene')
    .name('Tubes match scene')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(sceneDims, 'poolHalfExtent', 0.5, 8.0, 0.1)
    .name('Pool half width (X/Z)')
    .onChange(() => {
      syncSceneParams();
      pool.rebuildGeometry(sceneDims.poolHalfExtent);
      water.rebuildSurfaceMesh(sceneDims.poolHalfExtent);
      const r = effectiveObjectRadius();
      center.x = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.x)
      );
      center.z = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.z)
      );
      writeSphereUniforms(0);
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(sceneDims, 'poolDepth', 0.2, 5.0, 0.05)
    .name('Pool depth')
    .onChange(() => {
      syncSceneParams();
      const r = effectiveObjectRadius();
      if (center.y < r - sceneDims.poolDepth) {
        center.y = r - sceneDims.poolDepth;
        writeSphereUniforms(0);
      }
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(sceneDims, 'poolRimMaxY', 0.5, 10.0, 0.1)
    .name('Rim max Y')
    .onChange(() => {
      syncSceneParams();
      (document.activeElement as HTMLElement)?.blur();
    });

  sceneFolder
    .add(sceneDims, 'ballRadius', 0.05, 1.0, 0.01)
    .name('Ball radius')
    .onChange(() => {
      const r = effectiveObjectRadius();
      center.x = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.x)
      );
      center.y = Math.max(
        r - sceneDims.poolDepth,
        Math.min(10, center.y)
      );
      center.z = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.z)
      );
      writeSphereUniforms(0);
      (document.activeElement as HTMLElement)?.blur();
    });

  waterFolder
    .add(settings, 'causticsIntensity', 0.0, 1.0, 0.01)
    .name('Caustics')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  function applyLinerPreset(id: LinerPresetId): void {
    const p = LINER_PRESETS[id];
    linerAppearance.underTint[0] = p.underTint[0];
    linerAppearance.underTint[1] = p.underTint[1];
    linerAppearance.underTint[2] = p.underTint[2];
    linerAppearance.tileTint[0] = p.tileTint[0];
    linerAppearance.tileTint[1] = p.tileTint[1];
    linerAppearance.tileTint[2] = p.tileTint[2];
    linerAppearance.aboveTint[0] = p.aboveTint[0];
    linerAppearance.aboveTint[1] = p.aboveTint[1];
    linerAppearance.aboveTint[2] = p.aboveTint[2];
    sceneDims.waterAbsorption = p.waterAbsorption;
    syncSceneParams();
    depthAbsorptionGui?.updateDisplay();
  }

  waterFolder
    .add(settings, 'linerPreset', linerPresetLabels)
    .name('Liner preset')
    .onChange((label: string) => {
      const id = linerLabelToId.get(label);
      if (id) applyLinerPreset(id);
      (document.activeElement as HTMLElement)?.blur();
    });

  let depthAbsorptionGui: { updateDisplay: () => void } | undefined;

  depthAbsorptionGui = waterFolder
    .add(sceneDims, 'waterAbsorption', 0.0, 2.5, 0.05)
    .name('Depth absorption')
    .onChange(() => {
      syncSceneParams();
      (document.activeElement as HTMLElement)?.blur();
    });

  waterFolder
    .add(settings, 'ior', 1.0, 1.5, 0.001)
    .name('Refraction')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  waterFolder
    .add(settings, 'fresnelMin', 0.0, 1.0, 0.01)
    .name('Reflection')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  waterFolder
    .add(settings, 'surfaceRoughness', 0.0, 1.0, 0.01)
    .name('Surface roughness')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  waterFolder
    .add(settings, 'foamStrength', 0.0, 1.5, 0.01)
    .name('Foam')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  waveSimGui.inject = waveSimFolder
    .add(waveSim, 'sphereInject', 0.02, 0.28, 0.005)
    .name('Sphere injection')
    .onChange(() => {
      applyWaveSimulationParams();
      (document.activeElement as HTMLElement)?.blur();
    });
  waveSimGui.wave = waveSimFolder
    .add(waveSim, 'waveResponse', 0.22, MAX_WAVE_RESPONSE, 0.025)
    .name('Wave response')
    .onChange(() => {
      applyWaveSimulationParams();
      (document.activeElement as HTMLElement)?.blur();
    });
  waveSimGui.damp = waveSimFolder
    .add(waveSim, 'damping', 0.985, 0.9995, 0.0005)
    .name('Velocity damping')
    .onChange(() => {
      applyWaveSimulationParams();
      (document.activeElement as HTMLElement)?.blur();
    });

  applyWaveSimulationParams();

  // Initialize sphere position
  writeSphereUniforms(0);

  // Add initial random ripples
  for (let i = 0; i < 20; i++) {
    water.addDrop(Math.random() * 2 - 1, Math.random() * 2 - 1, 0.03, i & 1 ? 0.01 : -0.01);
  }

  // --- Keyboard Input ---

  /** Currently pressed keys (uppercase) */
  const keys: Record<string, boolean> = {};

  window.addEventListener('keydown', (e) => {
    const key = e.key.toUpperCase();
    keys[key] = true;
    if (key === 'G') {
      useSpherePhysics = !useSpherePhysics; // Toggle gravity
      settings.gravity = useSpherePhysics;
      gravityController.updateDisplay();
    } else if (key === ' ') paused = !paused; // Toggle pause
  });

  window.addEventListener('keyup', (e) => {
    keys[e.key.toUpperCase()] = false;
  });

  // --- Pointer Interaction (supports mouse, touch, and pen) ---

  /** Previous pointer X position */
  let oldX = 0;
  /** Previous pointer Y position */
  let oldY = 0;
  /** Previous hit point for sphere dragging */
  let prevHit: Vector;
  /** Plane normal for sphere dragging */
  let planeNormal: Vector;

  /** Active pointers for multi-touch handling */
  const activePointers = new Map<number, { x: number; y: number }>();
  /** Previous pinch distance for zoom calculation */
  let lastPinchDistance = 0;
  /** Midpoint of two touches — used to orbit when both fingers move together (not only pinch). */
  let lastTwoFingerCentroid: { x: number; y: number } | null = null;

  /** Last pointer position in client coordinates (for Tubes overlay + water hit-test). */
  let lastPointerClient: { x: number; y: number } | null = null;

  /**
   * Bitmask from the latest pointer event (`MouseEvent.buttons`). Tubes overlay only shows while
   * primary is held (`buttons & 1`): left mouse or touch contact.
   */
  let lastPointerButtons = 0;
  /** Last pointer device kind — touch needs a fallback when `buttons` stays 0 on move (see render). */
  let lastPointerType: string = 'mouse';

  /** Ripple injection when clicking/dragging the pool (stronger = heavier splashes). */
  const WATER_TOUCH_RADIUS = 0.044;
  const WATER_TOUCH_STRENGTH = 0.017;

  /**
   * One-frame multiplier for sphere→water displacement after a waterline cross (enter/exit).
   * Consumed inside {@link displaceWaterForSphereMotion}.
   */
  let splashDisplacementBoost = 1.0;

  /** Avoid rapid fire when oscillating across fSub threshold at the rim. */
  let lastWaterlineSplashMs = -Infinity;

  /**
   * Extra ripple + displacement when the ball crosses the air/water boundary (entry splash / exit).
   */
  function injectWaterlineSplash(
    fSubBefore: number,
    fSubAfter: number,
    vy: number,
    ballRadius: number,
    poolHalf: number,
    nowMs: number
  ): void {
    const threshold = 0.034;
    const entered = fSubBefore < threshold && fSubAfter >= threshold;
    const exited = fSubBefore >= threshold && fSubAfter < threshold;
    if (!entered && !exited) {
      return;
    }
    if (nowMs - lastWaterlineSplashMs < 220) {
      return;
    }
    lastWaterlineSplashMs = nowMs;
    const nx = center.x / poolHalf;
    const nz = center.z / poolHalf;
    if (Math.abs(nx) >= 0.992 || Math.abs(nz) >= 0.992) {
      return;
    }
    const vyAbs = Math.min(24, Math.abs(vy));
    const speedBoost = 0.52 + Math.min(1.55, vyAbs * 0.088);
    const dropR = Math.min(
      0.12,
      (0.036 + ballRadius * 0.55) * (entered ? 1.1 : 0.96)
    );
    const baseStr = 0.026;
    let strength = Math.min(0.12, (baseStr + vyAbs * 0.019) * speedBoost);
    if (exited) {
      strength *= 0.62;
    }
    water.addDrop(nx, nz, dropR, strength);
    splashDisplacementBoost = Math.max(
      splashDisplacementBoost,
      entered ? 1.68 : 1.42
    );
  }

  /**
   * Gets the current viewport as [x, y, width, height].
   */
  function getViewport(): Viewport {
    return [0, 0, canvas.width, canvas.height];
  }

  /**
   * Maps pointer position to canvas backing-store pixels (same space as {@link getViewport}).
   * Using bounding rect avoids mismatch between CSS pixels, devicePixelRatio, and floor() resize math.
   */
  function pointerToCanvasDevicePixels(clientX: number, clientY: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * sx,
      y: (clientY - rect.top) * sy,
    };
  }

  /** Ray vs horizontal plane y = planeY; returns null if parallel or hit behind the camera. */
  function intersectPlaneY(tracer: Raytracer, ray: Vector, planeY: number): Vector | null {
    if (Math.abs(ray.y) < 1e-6) return null;
    const t = (planeY - tracer.eye.y) / ray.y;
    if (!Number.isFinite(t) || t <= 0) return null;
    return tracer.eye.add(ray.multiply(t));
  }

  /** Distance along ray (same convention as {@link Raytracer.hitTestSphere}) to y = planeY. */
  function rayDistanceToPlaneY(tracer: Raytracer, ray: Vector, planeY: number): number | null {
    if (Math.abs(ray.y) < 1e-6) return null;
    const t = (planeY - tracer.eye.y) / ray.y;
    if (!Number.isFinite(t) || t <= 0) return null;
    return t;
  }

  /**
   * True if a screen-space pointer lies over the water surface (not blocked by the sphere).
   * Used to show the Tubes overlay whenever the cursor/touch aims at the pool.
   */
  function clientRayHitsWaterSurface(clientX: number, clientY: number): boolean {
    const { x, y } = pointerToCanvasDevicePixels(clientX, clientY);
    const { projectionMatrix, viewMatrix } = getMatrices();
    const tracer = new Raytracer(viewMatrix, projectionMatrix, getViewport());
    const ray = tracer.getRayForPixel(x, y);
    const sphereHit = objectIsSolid()
      ? Raytracer.hitTestSphere(tracer.eye, ray, center, effectiveObjectRadius())
      : null;
    const pointOnPlane = intersectPlaneY(tracer, ray, 0);
    if (!pointOnPlane) return false;
    if (
      Math.abs(pointOnPlane.x) >= sceneDims.poolHalfExtent ||
      Math.abs(pointOnPlane.z) >= sceneDims.poolHalfExtent
    ) {
      return false;
    }
    const tPlane = rayDistanceToPlaneY(tracer, ray, 0);
    if (tPlane === null) return false;
    if (sphereHit && sphereHit.t > 0 && sphereHit.t < tPlane) return false;
    return true;
  }

  /**
   * Handles pointer down - determines interaction mode.
   * @param x - Pointer X in canvas device pixels (0 to canvas.width)
   * @param y - Pointer Y in canvas device pixels (0 to canvas.height)
   * @param button - Pointer button (0=left, 2=right)
   * @param options.suppressInitialWaterDrop - After a pinch, resume water-drag without a splash on lift.
   */
  function startDrag(
    x: number,
    y: number,
    button: number,
    options?: { suppressInitialWaterDrop?: boolean }
  ): void {
    oldX = x;
    oldY = y;

    // Right click always orbits
    if (button === 2) {
      mode = InteractionMode.OrbitCamera;
      return;
    }

    const { projectionMatrix, viewMatrix } = getMatrices();
    const tracer = new Raytracer(viewMatrix, projectionMatrix, getViewport());
    const ray = tracer.getRayForPixel(x, y);

    // Check if clicking on sphere (only if visible)
    const sphereHit = objectIsSolid()
      ? Raytracer.hitTestSphere(tracer.eye, ray, center, effectiveObjectRadius())
      : null;
    if (sphereHit) {
      mode = InteractionMode.MoveSphere;
      prevHit = sphereHit.hit;
      // Use camera forward direction as drag plane normal
      planeNormal = tracer.getRayForPixel(canvas.width / 2, canvas.height / 2).negative();
      return;
    }

    // Check if clicking on water surface (y=0 plane)
    const pointOnPlane = intersectPlaneY(tracer, ray, 0);
    if (
      pointOnPlane &&
      Math.abs(pointOnPlane.x) < sceneDims.poolHalfExtent &&
      Math.abs(pointOnPlane.z) < sceneDims.poolHalfExtent
    ) {
      // Click is within water bounds (addDrop uses normalized [-1, 1] UV space)
      mode = InteractionMode.AddDrops;
      if (!options?.suppressInitialWaterDrop) {
        water.addDrop(
          pointOnPlane.x / sceneDims.poolHalfExtent,
          pointOnPlane.z / sceneDims.poolHalfExtent,
          WATER_TOUCH_RADIUS,
          WATER_TOUCH_STRENGTH
        );
      }
    } else {
      // Click is outside water - orbit camera
      mode = InteractionMode.OrbitCamera;
    }
  }

  /** When lifting one finger after a two-finger pinch, the remaining touch must get a fresh hit-test. */
  function resumeSinglePointerAfterPinch(): void {
    if (activePointers.size !== 1) return;
    const p = activePointers.values().next().value;
    if (!p) return;
    startDrag(p.x, p.y, 0, { suppressInitialWaterDrop: true });
  }

  /**
   * Handles pointer move during drag.
   * @param x - Current pointer X in canvas device pixels
   * @param y - Current pointer Y in canvas device pixels
   */
  function duringDrag(x: number, y: number): void {
    if (mode === InteractionMode.OrbitCamera) {
      // Rotate camera based on pointer delta
      targetAngleY -= x - oldX;
      targetAngleX -= y - oldY;
      targetAngleX = Math.max(-89.999, Math.min(89.999, targetAngleX)); // Clamp pitch
    } else if (mode === InteractionMode.MoveSphere) {
      // Move sphere along drag plane
      const { projectionMatrix, viewMatrix } = getMatrices();
      const tracer = new Raytracer(viewMatrix, projectionMatrix, getViewport());
      const ray = tracer.getRayForPixel(x, y);

      // Intersect ray with drag plane
      const t = -planeNormal.dot(tracer.eye.subtract(prevHit)) / planeNormal.dot(ray);
      const nextHit = tracer.eye.add(ray.multiply(t));

      // Update sphere position with bounds checking
      center = center.add(nextHit.subtract(prevHit));
      const r = effectiveObjectRadius();
      center.x = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.x)
      );
      center.y = Math.max(
        r - sceneDims.poolDepth,
        Math.min(10, center.y)
      );
      center.z = Math.max(
        r - sceneDims.poolHalfExtent,
        Math.min(sceneDims.poolHalfExtent - r, center.z)
      );

      writeSphereUniforms(0);
      prevHit = nextHit;
    } else if (mode === InteractionMode.AddDrops) {
      // Add ripples while dragging on water
      const { projectionMatrix, viewMatrix } = getMatrices();
      const tracer = new Raytracer(viewMatrix, projectionMatrix, getViewport());
      const ray = tracer.getRayForPixel(x, y);
      const pointOnPlane = intersectPlaneY(tracer, ray, 0);

      if (
        pointOnPlane &&
        Math.abs(pointOnPlane.x) < sceneDims.poolHalfExtent &&
        Math.abs(pointOnPlane.z) < sceneDims.poolHalfExtent
      ) {
        water.addDrop(
          pointOnPlane.x / sceneDims.poolHalfExtent,
          pointOnPlane.z / sceneDims.poolHalfExtent,
          WATER_TOUCH_RADIUS,
          WATER_TOUCH_STRENGTH
        );
      }
    }
    oldX = x;
    oldY = y;
  }

  /**
   * Handles pointer up - ends interaction and releases capture.
   */
  function stopDrag(): void {
    mode = InteractionMode.None;
  }

  /**
   * Calculates the distance between two pointers (for pinch-to-zoom).
   */
  function getPinchDistance(): number {
    const pointers = Array.from(activePointers.values());
    if (pointers.length < 2) return 0;
    const dx = pointers[0].x - pointers[1].x;
    const dy = pointers[0].y - pointers[1].y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function getTwoFingerCentroid(): { x: number; y: number } | null {
    if (activePointers.size < 2) return null;
    const pointers = Array.from(activePointers.values());
    return {
      x: (pointers[0].x + pointers[1].x) * 0.5,
      y: (pointers[0].y + pointers[1].y) * 0.5,
    };
  }

  // Pointer event listeners (unified mouse/touch/pen input)
  // Capture phase: TubesCursor / overlay may stop bubbling; we still need coords for water hit-test + neon overlay.
  const syncLastPointerForTubes = (e: PointerEvent) => {
    lastPointerButtons = e.buttons;
    lastPointerType = e.pointerType;
    const t = e.target;
    if (t instanceof Node && viewArea.contains(t)) {
      lastPointerClient = { x: e.clientX, y: e.clientY };
    }
  };
  window.addEventListener('pointermove', syncLastPointerForTubes, true);
  window.addEventListener('pointerdown', syncLastPointerForTubes, true);
  window.addEventListener('pointerup', syncLastPointerForTubes, true);
  window.addEventListener('pointercancel', syncLastPointerForTubes, true);
  viewArea.addEventListener('pointerleave', () => {
    lastPointerClient = null;
  });

  canvas.addEventListener('pointerdown', (e) => {
    if (e.button === 1) return; // Ignore middle click
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);

    // Track this pointer (device pixels - consistent with Raytracer viewport)
    const p = pointerToCanvasDevicePixels(e.clientX, e.clientY);
    activePointers.set(e.pointerId, p);

    // If this is the second finger, switch to pinch + two-finger orbit (centroid drag)
    if (activePointers.size === 2) {
      mode = InteractionMode.None; // Cancel any single-finger interaction
      lastPinchDistance = getPinchDistance();
      lastTwoFingerCentroid = getTwoFingerCentroid();
      return;
    }

    // Only start drag interaction if this is the first/only pointer
    if (activePointers.size === 1) {
      startDrag(p.x, p.y, e.button);
    }
  });

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener(
    'pointermove',
    (e) => {
      const p = pointerToCanvasDevicePixels(e.clientX, e.clientY);
      // Reduce browser gestures (scroll/pull-to-refresh) stealing the gesture while dragging or pinching.
      if (mode !== InteractionMode.None || activePointers.size >= 2) {
        e.preventDefault();
      }
      // Update pointer position in our tracking map
      if (activePointers.has(e.pointerId)) {
        activePointers.set(e.pointerId, p);
      }

      // Two fingers: orbit from centroid movement + pinch-to-zoom from distance change
      if (activePointers.size === 2) {
        const c = getTwoFingerCentroid();
        if (c && lastTwoFingerCentroid) {
          targetAngleY -= c.x - lastTwoFingerCentroid.x;
          targetAngleX -= c.y - lastTwoFingerCentroid.y;
          targetAngleX = Math.max(-89.999, Math.min(89.999, targetAngleX));
        }
        if (c) {
          lastTwoFingerCentroid = c;
        }
        const currentDistance = getPinchDistance();
        if (lastPinchDistance > 0) {
          const delta = lastPinchDistance - currentDistance;
          targetDistance += delta * 0.01;
          targetDistance = Math.max(1.5, Math.min(10, targetDistance));
        }
        lastPinchDistance = currentDistance;
        return;
      }

      // Single pointer drag
      if (mode !== InteractionMode.None && activePointers.size === 1) {
        duringDrag(p.x, p.y);
      }
    },
    { passive: false }
  );

  canvas.addEventListener('pointerup', (e) => {
    const countBefore = activePointers.size;
    canvas.releasePointerCapture(e.pointerId);
    activePointers.delete(e.pointerId);

    // Reset pinch / two-finger orbit state
    if (activePointers.size < 2) {
      lastPinchDistance = 0;
      lastTwoFingerCentroid = null;
    }

    // Pinch ended but one finger still down — otherwise mode stays None until retouch (mobile bug).
    if (countBefore === 2 && activePointers.size === 1) {
      resumeSinglePointerAfterPinch();
    }

    // Only fully stop drag when all pointers are released
    if (activePointers.size === 0) {
      stopDrag();
    }
  });

  canvas.addEventListener('pointercancel', (e) => {
    const countBefore = activePointers.size;
    canvas.releasePointerCapture(e.pointerId);
    activePointers.delete(e.pointerId);

    if (activePointers.size < 2) {
      lastPinchDistance = 0;
      lastTwoFingerCentroid = null;
    }

    if (countBefore === 2 && activePointers.size === 1) {
      resumeSinglePointerAfterPinch();
    }

    if (activePointers.size === 0) {
      stopDrag();
    }
  });

  // Wheel event for zooming
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      targetDistance += e.deltaY * 0.005;
      targetDistance = Math.max(1.5, Math.min(10, targetDistance));
    },
    { passive: false }
  );

  // --- Rendering ---

  /** Depth texture for 3D rendering */
  let depthTexture: GPUTexture;

  /**
   * Handles window resize - updates canvas size and recreates depth texture.
   */
  function onResize(): void {
    const isMobile = window.matchMedia('(max-width: 600px)').matches;
    const helpCollapsed = help.classList.contains('collapsed');
    // Mobile: help overlay full width. Desktop: reserve space only when help panel is open.
    const width = isMobile
      ? window.innerWidth
      : helpCollapsed
        ? window.innerWidth
        : window.innerWidth - help.clientWidth - 20;
    const height = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    viewArea.style.width = `${width}px`;
    viewArea.style.height = `${height}px`;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    syncTubesCanvasSize(tubesCanvas, canvas);
    resizeTubesCursorHost(tubesApp);

    // Recreate depth texture at new size
    if (depthTexture) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    render();
  }

  window.addEventListener('resize', onResize);

  const helpToggle = document.getElementById('help-toggle')!;
  helpToggle.addEventListener('click', () => {
    help.classList.toggle('collapsed');
    helpToggle.textContent = help.classList.contains('collapsed') ? 'menu' : 'chevron_right';
    onResize();
  });

  // Collapse help panel when clicking outside of it (mobile only)
  window.addEventListener('pointerdown', (e) => {
    const isMobile = window.matchMedia('(max-width: 600px)').matches;
    if (isMobile && !help.classList.contains('collapsed')) {
      const target = e.target as HTMLElement;
      if (!help.contains(target) && !helpToggle.contains(target)) {
        help.classList.add('collapsed');
        helpToggle.textContent = 'menu';
        onResize();
      }
    }
  });

  document.getElementById('loading')!.style.display = 'none';
  document.body.classList.remove('loading');
  onResize();
  // TubesCursor BC.resize() uses parent #view-area size — must run after first onResize.
  tubesApp = initTubesCursor(tubesCanvas);
  resizeTubesCursorHost(tubesApp);
  // Library listens on the overlay canvas; pointer-events must reach it (then clone to scene).
  wireOverlayPointerPassthrough(tubesCanvas, canvas);

  /**
   * Projects sphere motion into the wave heightfield. Fast bounces inject huge per-frame
   * deltas; without scaling, the wave integrator can diverge (NaNs → black frame / GPU TDR).
   * GPU passes also clamp sim height/velocity; we soften injection when speed is high.
   */
  function displaceWaterForSphereMotion(dt: number): void {
    const r = effectiveObjectRadius();
    const speed =
      dt > 1e-8
        ? Math.hypot(
            center.x - oldCenter.x,
            center.y - oldCenter.y,
            center.z - oldCenter.z
          ) / dt
        : 0;
    const injectAtten = Math.min(1 / (1 + speed * 0.42), 0.92);
    const splashBoost = splashDisplacementBoost;
    splashDisplacementBoost = 1.0;
    const { waveResponseStep, dampingStep } = computeWaveStepUniforms();
    water.setSimulationParams(
      waveSim.sphereInject * injectAtten * splashBoost,
      waveResponseStep,
      dampingStep
    );
    water.moveSphere(oldCenter.toArray(), center.toArray(), r);
    water.setSimulationParams(
      waveSim.sphereInject,
      waveResponseStep,
      dampingStep
    );
  }

  /**
   * Updates the common uniform buffer with current camera matrices.
   * @returns Camera eye Y in world space (for TubesCursor underwater fade).
   */
  function updateUniforms(): { eyeY: number } {
    const { projectionMatrix, viewMatrix } = getMatrices();
    const viewProjectionMatrix = mat4.multiply(projectionMatrix, viewMatrix);

    // Calculate eye position from inverse view matrix
    const invView = mat4.invert(viewMatrix);
    const eyeVec = vec3.transformMat4([0, 0, 0], invView);

    // Pack into uniform buffer: mat4 (16 floats) + vec3 (3 floats) + padding (1 float)
    const uniformData = new Float32Array(20);
    uniformData.set(viewProjectionMatrix, 0);
    uniformData.set(eyeVec, 16);

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    return { eyeY: eyeVec[1] };
  }

  /**
   * Main render function - called every frame.
   */
  function render(): void {
    // Calculate delta time
    const time = performance.now();
    let seconds = (time - prevTime) / 1000;
    prevTime = time;
    if (seconds > 1) seconds = 1; // Cap delta time to prevent physics explosion

    // Smoothly interpolate camera towards targets (damping)
    angleX += (targetAngleX - angleX) * 0.15;
    angleY += (targetAngleY - angleY) * 0.15;
    distance += (targetDistance - distance) * 0.15;

    // Update light direction if L key is held or Follow Camera is enabled
    if (keys['L'] || settings.followCamera) {
      lightDir = Vector.fromAngles(((90 - angleY) * Math.PI) / 180, (-angleX * Math.PI) / 180);
      updateLight();
    }

    // Update water rendering uniforms (see WaterUniforms in bindings.wgsl)
    // Do this before simulation steps so caustics update uses correct values
    water.updateWaterParameters(
      settings.useDensity ? effectiveObjectDensity() : 0.0,
      settings.causticsIntensity,
      settings.ior,
      settings.fresnelMin,
      settings.surfaceRoughness,
      settings.foamStrength,
      water.getWaterTexel()
    );

    if (!paused) {
      // --- Physics Update ---

      if (mode === InteractionMode.MoveSphere) {
        // User is dragging sphere - stop physics
        velocity = new Vector();
      } else if (useSpherePhysics) {
        const r = effectiveObjectRadius();
        const waveCpu = water.getWaveCpuSample();
        const useWaveCoupling = settings.waveBodyCoupling && waveCpu.valid;
        const surfacePlaneY = useWaveCoupling ? waveCpu.surfaceWorldY : 0;
        // Fraction of sphere volume below the local water plane (flat surface at y=0 when coupling off)
        const fSubBefore = useWaveCoupling
          ? submergedVolumeFractionBelowPlane(center.y, r, surfacePlaneY)
          : submergedVolumeFraction(center.y, r);
        const fSub = fSubBefore;
        const buoy = buoyancyStrength(effectiveObjectDensity(), settings.useDensity);

        // Net vertical acceleration: gravity minus buoyancy proportional to submerged volume
        const g = -15.0;
        velocity.y += g * (1.0 - buoy * fSub) * seconds;

        // Quadratic fluid drag (stronger when more volume is submerged)
        if (velocity.length() > 1e-8) {
          const dragMag = fSub * seconds * velocity.dot(velocity) * 2.4;
          velocity = velocity.subtract(velocity.unit().multiply(dragMag));
        }

        // Air resistance on the out-of-water portion (approximated by 1 - submerged fraction)
        const airResistance = 0.11;
        const inAir = Math.max(0, 1 - fSub);
        velocity = velocity.multiply(1.0 - airResistance * seconds * inAir);

        // Ride waves + slide along slope (GPU heightfield sample lags ~1 frame)
        if (useWaveCoupling) {
          // Blend linear + quadratic submergence: react clearly when partly afloat, without rim twitch.
          const couplingWeight = Math.max(
            0,
            Math.min(1, fSub * 0.55 + fSub * fSub * 0.85)
          );
          // Fixed offset (surface − 0.32r) pulls light shells too deep vs Archimedes equilibrium
          // (ρ≈0.13 ⇒ ~13% submerged). That fights buoyancy → endless bob. Match float depth to ρ.
          let targetCenterY: number;
          if (settings.useDensity) {
            const rhoObj = effectiveObjectDensity();
            if (rhoObj <= 0.995) {
              const fEq = Math.min(0.97, Math.max(0.05, rhoObj));
              targetCenterY = centerYForSubmergedFractionBelowPlane(
                waveCpu.surfaceWorldY,
                r,
                fEq
              );
            } else {
              targetCenterY = waveCpu.surfaceWorldY - r * 0.36;
            }
          } else {
            targetCenterY = waveCpu.surfaceWorldY - r * 0.32;
          }
          // Softer spring for very light objects — avoids hunting when waves lag the target by a frame.
          const rhoForSpring = settings.useDensity ? effectiveObjectDensity() : 0.55;
          const lightSpringAtten =
            rhoForSpring < 0.5 ? 0.18 + 0.80 * rhoForSpring : 1.0;
          velocity.y +=
            WAVE_RIDE_SPRING *
            lightSpringAtten *
            (targetCenterY - center.y) *
            couplingWeight *
            seconds;
          // Vertical velocity damping near the surface — kills lag-induced bob without a PD spring.
          const rho = settings.useDensity ? effectiveObjectDensity() : 0.55;
          const lightVyBoost = Math.max(0, 0.5 - rho) * 10;
          const surfaceBlend = Math.min(1, fSub * 2.4 + 0.08);
          const vyDamp = (5.2 + lightVyBoost) * surfaceBlend * seconds;
          velocity.y *= Math.max(0.38, 1 - Math.min(0.78, vyDamp));
          // Slightly stronger than fSub² so swell drift is felt before full submergence.
          const slopePush = Math.pow(Math.max(0, Math.min(1, fSub)), 1.35);
          const sx = Math.max(
            -WAVE_MAX_SURFACE_SLOPE,
            Math.min(WAVE_MAX_SURFACE_SLOPE, waveCpu.dhdxWorld)
          );
          const sz = Math.max(
            -WAVE_MAX_SURFACE_SLOPE,
            Math.min(WAVE_MAX_SURFACE_SLOPE, waveCpu.dhdzWorld)
          );
          velocity.x += WAVE_SLOPE_ACCEL * sx * slopePush * seconds;
          velocity.z += WAVE_SLOPE_ACCEL * sz * slopePush * seconds;
        }

        // Surface interaction damping near the local water plane (splash / bob). When wave coupling
        // is on, the plane moves with the wave — full damping vs that plane erodes wave-driven motion;
        // scale down so bob/splash damp stays without killing swell drift.
        const effectiveDensity = settings.useDensity
          ? effectiveObjectDensity()
          : 1.0;
        const distanceFromSurface = Math.abs(center.y - surfacePlaneY);
        const surfaceProximity = Math.max(0, 1 - distanceFromSurface / r);
        const baseDamping = 0.48;
        const densityDamping = 0.48 * effectiveDensity;
        const surfaceDampScale = useWaveCoupling ? 0.4 : 1.0;
        const surfaceDamping =
          1.0 -
          surfaceProximity *
            (baseDamping + densityDamping) *
            seconds *
            surfaceDampScale;
        velocity = velocity.multiply(Math.max(0, surfaceDamping));

        center = center.add(velocity.multiply(seconds));

        resolvePoolCollisions(center, velocity, r, {
          halfExtent: sceneDims.poolHalfExtent,
          depth: sceneDims.poolDepth,
        });

        const fSubAfter = useWaveCoupling
          ? submergedVolumeFractionBelowPlane(center.y, r, surfacePlaneY)
          : submergedVolumeFraction(center.y, r);
        injectWaterlineSplash(
          fSubBefore,
          fSubAfter,
          velocity.y,
          r,
          sceneDims.poolHalfExtent,
          time
        );
      }

      if (objectIsSolid()) {
        displaceWaterForSphereMotion(seconds);
      }
      oldCenter = center.clone();

      // Wave heightfield: several substeps/frame; GPU uses 5-neighbor average + mirrored rim + curvature clamp.
      {
        const { waveResponseStep, dampingStep } = computeWaveStepUniforms();
        water.setSimulationParams(waveSim.sphereInject, waveResponseStep, dampingStep);
        for (let s = 0; s < WAVE_SIM_SUBSTEPS; s++) {
          water.stepSimulation();
        }
      }
    } else if (mode === InteractionMode.MoveSphere && objectIsSolid()) {
      // Simulation paused but user is dragging the sphere - still apply displacement so ripples show
      displaceWaterForSphereMotion(seconds);
      oldCenter = center.clone();
    }

    // Always derive normals + caustics from the current height texture - even when paused.
    // Otherwise addDrop/clicks while paused update height but lighting stays stale until resume.
    water.updateNormals();
    water.updateCaustics();

    if (
      settings.waveBodyCoupling &&
      objectIsSolid() &&
      mode !== InteractionMode.MoveSphere
    ) {
      water.queueWaveHeightReadback(center.x, center.z);
    }

    // UFO spin + sphere uniforms (every frame so drag/pause/GUI stay in sync)
    writeSphereUniforms(seconds);

    // Update camera uniforms
    const { eyeY } = updateUniforms();
    const pointerOverWater =
      lastPointerClient !== null &&
      clientRayHitsWaterSurface(lastPointerClient.x, lastPointerClient.y);
    /** Neon tubes only while primary is pressed (left button / touch contact), not on hover. */
    const tubesPrimaryHeld =
      (lastPointerButtons & 1) !== 0 ||
      (lastPointerType === 'touch' &&
        (mode === InteractionMode.AddDrops ||
          mode === InteractionMode.MoveSphere ||
          mode === InteractionMode.OrbitCamera));
    const waterContact =
      tubesPrimaryHeld && (mode === InteractionMode.AddDrops || pointerOverWater);
    const pointerPresenceScale =
      mode === InteractionMode.AddDrops ? 1 : pointerOverWater ? 0.68 : 1;

    let grazingFactor = 0;
    if (pointerOverWater && lastPointerClient && tubesPrimaryHeld) {
      const p = pointerToCanvasDevicePixels(lastPointerClient.x, lastPointerClient.y);
      const { projectionMatrix, viewMatrix } = getMatrices();
      const tracer = new Raytracer(viewMatrix, projectionMatrix, getViewport());
      const rd = tracer.getRayForPixel(p.x, p.y);
      grazingFactor = Math.min(1, Math.hypot(rd.x, rd.z));
    }

    const lightHueRotateDeg =
      settings.tubesMatchScene && settings.cursorTubes
        ? (Math.atan2(lightDir.x, lightDir.z) * 180) / Math.PI
        : 0;
    const overlayHueRotateDeg = lightHueRotateDeg * 0.07;

    syncTubesCursorScenePalette({
      app: tubesApp,
      enabled: settings.cursorTubes,
      sceneIntegration: settings.tubesMatchScene,
      lightDir: { x: lightDir.x, y: lightDir.y, z: lightDir.z },
      causticsIntensity: settings.causticsIntensity,
    });

    updateTubesOverlayStyle(tubesCanvas, {
      enabled: settings.cursorTubes,
      waterContact,
      eyeY,
      underwaterMinOpacity: settings.tubesUnderwaterMinOpacity,
      grazingFactor,
      lightHueRotateDeg: overlayHueRotateDeg,
      pointerPresenceScale,
    });

    // --- GPU Render Pass ---

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    // Render scene objects
    pool.render(passEncoder, water.textureA, water.sampler, water.causticsTexture);
    if (objectIsSolid()) {
      sphere.render(passEncoder, water.textureA, water.sampler, water.causticsTexture);
    }
    water.renderSurface(passEncoder);

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Animation loop - calls render every frame.
   */
  function animate(): void {
    requestAnimationFrame(animate);
    render();
  }

  // Start the animation loop
  requestAnimationFrame(animate);
}

// Initialize the application
init();
