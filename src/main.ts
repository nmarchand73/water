/**
 * main.ts - WebGPU Water Simulation Entry Point
 *
 * This is the main entry point for the interactive water simulation demo.
 * It initializes WebGPU, loads resources, sets up event handlers, and runs
 * the main render loop.
 *
 * Features:
 * - Interactive water ripples (click/drag on water surface)
 * - Draggable sphere with physics (gravity, buoyancy)
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
} from './scene-constants';
import ufoObjUrl from './shapes/UFO_Saucer.obj?url';
import { fetchAndParseUfoObj } from './shapes/parse-ufo-obj';

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
  const canvas = document.querySelector('canvas')!;
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

  // SphereUniforms: center, radius, spinY, shapeKind, pad (32 bytes - see bindings.wgsl)
  const sphereUniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Shadow toggle flags (3 floats + padding = 16 bytes)
  const shadowUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Water rendering uniforms (density, causticIntensity, ior, fresnelMin)
  const waterUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Pool / scene scale for GPU (`SceneParams`: halfExtent, depth, rimMaxY, pad)
  const sceneParamsBuffer = device.createBuffer({
    label: 'SceneParams',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneDims = {
    poolHalfExtent: DEFAULT_POOL_HALF_EXTENT,
    poolDepth: DEFAULT_POOL_DEPTH,
    poolRimMaxY: DEFAULT_POOL_RIM_MAX_Y,
    ballRadius: DEFAULT_BALL_RADIUS,
  };

  function syncSceneParams(): void {
    device.queue.writeBuffer(
      sceneParamsBuffer,
      0,
      new Float32Array([
        sceneDims.poolHalfExtent,
        sceneDims.poolDepth,
        sceneDims.poolRimMaxY,
        0,
      ])
    );
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
    waveResponse: 1.0,
    damping: 0.997,
  };
  waveSim.waveResponse = Math.min(waveSim.waveResponse, MAX_WAVE_RESPONSE);

  const waveSimGui: {
    inject?: { updateDisplay: () => void };
    wave?: { updateDisplay: () => void };
    damp?: { updateDisplay: () => void };
  } = {};

  function applyWaveSimulationParams(): void {
    water.setSimulationParams(waveSim.sphereInject, waveSim.waveResponse, waveSim.damping);
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

  /** Accumulated Y rotation for UFO mesh (radians). */
  let ufoSpin = 0;

  // --- GUI ---
  const gui = new GUI({ title: 'Settings' });
  gui.close(); // Collapse by default

  const waterFolder = gui.addFolder('Water');
  const objectFolder = gui.addFolder('Object');
  const sceneFolder = gui.addFolder('Scene');
  const waveSimFolder = gui.addFolder('Wave simulation');

  const settings = {
    gravity: useSpherePhysics,
    followCamera: false,
    object: 'Sphere',
    useDensity: false,
    density: 0.9,
    causticsIntensity: 0.2,
    ior: 1.333,
    fresnelMin: 0.25,
  };

  const gravityController = objectFolder
    .add(settings, 'gravity')
    .name('Toggle Gravity')
    .onChange((v: boolean) => {
      useSpherePhysics = v;
      (document.activeElement as HTMLElement)?.blur();
    });

  objectFolder
    .add(settings, 'useDensity')
    .name('Enable Density')
    .onChange(() => {
      updateDensityVisibility();
      (document.activeElement as HTMLElement)?.blur();
    });

  const densitySlider = objectFolder
    .add(settings, 'density', 0.2, 2.0, 0.1)
    .name('Density')
    .onChange(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

  /**
   * Shows/hides density slider based on useDensity setting.
   */
  function updateDensityVisibility(): void {
    densitySlider.show(settings.useDensity);
  }

  // Initialize density control visibility
  updateDensityVisibility();

  /** Visual + physics + water radius for the current object (UFO is scaled up vs sphere). */
  function effectiveObjectRadius(): number {
    return settings.object === 'UFO'
      ? sceneDims.ballRadius * UFO_RADIUS_SCALE
      : sceneDims.ballRadius;
  }

  /** Writes sphere/UFO GPU uniforms; advances UFO spin when `dtSpin > 0` and object is UFO. */
  function writeSphereUniforms(dtSpin: number): void {
    if (settings.object === 'UFO' && dtSpin > 0 && !paused) {
      ufoSpin += dtSpin * 2.4;
    }
    sphere.setMeshKind(settings.object === 'UFO' ? 'ufo' : 'sphere');
    const shapeKind = settings.object === 'UFO' ? 1 : 0;
    const spinY = settings.object === 'UFO' ? ufoSpin : 0;
    sphere.update(center.toArray(), effectiveObjectRadius(), spinY, shapeKind);
  }

  function objectIsSolid(): boolean {
    return settings.object === 'Sphere' || settings.object === 'UFO';
  }

  objectFolder
    .add(settings, 'object', ['Sphere', 'UFO', 'None'])
    .name('Object')
    .onChange((v: string) => {
      const isVisible = v === 'Sphere' || v === 'UFO';
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

  waveSimGui.inject = waveSimFolder
    .add(waveSim, 'sphereInject', 0.02, 0.28, 0.005)
    .name('Sphere injection')
    .onChange(() => {
      applyWaveSimulationParams();
      (document.activeElement as HTMLElement)?.blur();
    });
  waveSimGui.wave = waveSimFolder
    .add(waveSim, 'waveResponse', 0.4, MAX_WAVE_RESPONSE, 0.05)
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

  /** Current interaction mode */
  let mode: InteractionMode = InteractionMode.None;
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

  /**
   * Handles pointer down - determines interaction mode.
   * @param x - Pointer X in canvas device pixels (0 to canvas.width)
   * @param y - Pointer Y in canvas device pixels (0 to canvas.height)
   * @param button - Pointer button (0=left, 2=right)
   */
  function startDrag(x: number, y: number, button: number): void {
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
      water.addDrop(
        pointOnPlane.x / sceneDims.poolHalfExtent,
        pointOnPlane.z / sceneDims.poolHalfExtent,
        0.03,
        0.01
      );
    } else {
      // Click is outside water - orbit camera
      mode = InteractionMode.OrbitCamera;
    }
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
          0.03,
          0.01
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

  // Pointer event listeners (unified mouse/touch/pen input)
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button === 1) return; // Ignore middle click
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);

    // Track this pointer (device pixels - consistent with Raytracer viewport)
    const p = pointerToCanvasDevicePixels(e.clientX, e.clientY);
    activePointers.set(e.pointerId, p);

    // If this is the second finger, switch to pinch mode and record initial distance
    if (activePointers.size === 2) {
      mode = InteractionMode.None; // Cancel any single-finger interaction
      lastPinchDistance = getPinchDistance();
      return;
    }

    // Only start drag interaction if this is the first/only pointer
    if (activePointers.size === 1) {
      startDrag(p.x, p.y, e.button);
    }
  });

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener('pointermove', (e) => {
    const p = pointerToCanvasDevicePixels(e.clientX, e.clientY);
    // Update pointer position in our tracking map
    if (activePointers.has(e.pointerId)) {
      activePointers.set(e.pointerId, p);
    }

    // Handle pinch-to-zoom with two fingers
    if (activePointers.size === 2) {
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
  });

  canvas.addEventListener('pointerup', (e) => {
    canvas.releasePointerCapture(e.pointerId);
    activePointers.delete(e.pointerId);

    // Reset pinch state
    if (activePointers.size < 2) {
      lastPinchDistance = 0;
    }

    // Only fully stop drag when all pointers are released
    if (activePointers.size === 0) {
      stopDrag();
    }
  });

  canvas.addEventListener('pointercancel', (e) => {
    canvas.releasePointerCapture(e.pointerId);
    activePointers.delete(e.pointerId);

    if (activePointers.size < 2) {
      lastPinchDistance = 0;
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
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

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

  /**
   * Updates the common uniform buffer with current camera matrices.
   */
  function updateUniforms(): void {
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

    // Update water rendering uniforms (density, caustics, IOR, fresnel)
    // Do this before simulation steps so caustics update uses correct values
    water.updateWaterParameters(
      settings.useDensity ? settings.density : 0.0,
      settings.causticsIntensity,
      settings.ior,
      settings.fresnelMin
    );

    if (!paused) {
      // --- Physics Update ---

      if (mode === InteractionMode.MoveSphere) {
        // User is dragging sphere - stop physics
        velocity = new Vector();
      } else if (useSpherePhysics) {
        // Apply gravity and buoyancy
        const r = effectiveObjectRadius();
        const percentUnderWater = Math.max(
          0,
          Math.min(
            1,
            (r - center.y) / (2 * r)
          )
        );

        // Buoyancy factor: 1/density when density enabled, otherwise default 1.1
        const buoyancyFactor = settings.useDensity ? 1.0 / settings.density : 1.1;

        // Gravity and buoyancy (using -15 for more snappy "fast" physics)
        const g = -15.0;
        velocity.y += (g - buoyancyFactor * g * percentUnderWater) * seconds;

        // Water drag proportional to velocity squared (when underwater)
        if (velocity.length() > 0) {
          velocity = velocity.subtract(
            velocity.unit().multiply(percentUnderWater * seconds * velocity.dot(velocity) * 2.0)
          );
        }

        // Air resistance (much smaller, time-dependent damping)
        const airResistance = 0.1; // 10% velocity loss per second
        const aboveWaterFactor = 1 - percentUnderWater;
        velocity = velocity.multiply(1.0 - airResistance * seconds * aboveWaterFactor);

        // Surface damping - energy loss when crossing water surface (splashing)
        // Normalized to be frame-rate independent
        const effectiveDensity = settings.useDensity ? settings.density : 1.0;
        const distanceFromSurface = Math.abs(center.y);
        const surfaceProximity = Math.max(
          0,
          1 - distanceFromSurface / r
        );
        const baseDamping = 0.5; // Damping rate per second
        const densityDamping = 0.5 * effectiveDensity;
        const surfaceDamping = 1.0 - surfaceProximity * (baseDamping + densityDamping) * seconds;
        velocity = velocity.multiply(Math.max(0, surfaceDamping));

        center = center.add(velocity.multiply(seconds));

        // Floor collision
        if (center.y < r - sceneDims.poolDepth) {
          center.y = r - sceneDims.poolDepth;
          velocity.y = Math.abs(velocity.y) * 0.7; // Bounce with energy loss
        }
      }

      if (objectIsSolid()) {
        // Update water displacement from sphere movement
        water.moveSphere(oldCenter.toArray(), center.toArray(), effectiveObjectRadius());
      }
      oldCenter = center.clone();

      // Run water simulation (twice per frame for smoother waves)
      water.stepSimulation();
      water.stepSimulation();
    } else if (mode === InteractionMode.MoveSphere && objectIsSolid()) {
      // Simulation paused but user is dragging the sphere - still apply displacement so ripples show
      water.moveSphere(oldCenter.toArray(), center.toArray(), effectiveObjectRadius());
      oldCenter = center.clone();
    }

    // Always derive normals + caustics from the current height texture - even when paused.
    // Otherwise addDrop/clicks while paused update height but lighting stays stale until resume.
    water.updateNormals();
    water.updateCaustics();

    // UFO spin + sphere uniforms (every frame so drag/pause/GUI stay in sync)
    writeSphereUniforms(seconds);

    // Update camera uniforms
    updateUniforms();

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
