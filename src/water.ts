/**
 * water.ts - Water Simulation and Rendering System
 *
 * This is the core module that implements the interactive water simulation.
 * It handles both the physics simulation and the visual rendering of water.
 *
 * The simulation uses a heightfield approach where water height is stored
 * in a 2D texture. The simulation runs on the GPU using render passes.
 *
 * Key components:
 * - Physics simulation: Wave propagation using neighboring height values
 * - Drop creation: Adding ripples from user interaction
 * - Sphere interaction: Water displacement from moving sphere
 * - Normal calculation: Computing surface normals for lighting
 * - Surface rendering: Reflections and refractions using ray tracing
 * - Caustics: Light patterns on pool floor from water surface refraction
 */

import type { PipelineConfig } from './types';
import { REF_POOL_HALF_EXTENT } from './scene-constants';
import {
  waterHeightWorld,
  type WaveCpuSample,
  EMPTY_WAVE_CPU_SAMPLE,
} from './water-cpu';

// Import shader modules
import fullscreenVertShader from './shaders/water/fullscreen.vert.wgsl';
import dropFragShader from './shaders/water/drop.frag.wgsl';
import updateFragShader from './shaders/water/update.frag.wgsl';
import normalFragShader from './shaders/water/normal.frag.wgsl';
import sphereFragShader from './shaders/water/sphere.frag.wgsl';
import surfaceVertShader from './shaders/water/surface.vert.wgsl';
import surfaceAboveFragShader from './shaders/water/surface-above.frag.wgsl';
import surfaceUnderFragShader from './shaders/water/surface-under.frag.wgsl';
import causticsVertShader from './shaders/water/caustics.vert.wgsl';
import causticsFragShader from './shaders/water/caustics.frag.wgsl';

/**
 * Main water simulation and rendering class.
 *
 * The Water class manages:
 * 1. Two ping-pong textures for double-buffered simulation
 * 2. Multiple compute pipelines for different simulation steps
 * 3. Surface mesh for rendering the water from above and below
 * 4. Caustics texture for underwater light patterns
 */
export class Water {
  /** WebGPU device for all GPU operations */
  private device: GPUDevice;

  /** Width of the simulation texture in pixels */
  private width: number;

  /** Height of the simulation texture in pixels */
  private height: number;

  // --- External Resources ---
  // These buffers and textures are passed in from main.ts

  /** Common uniform buffer (view-projection matrix, eye position) */
  private commonUniformBuffer: GPUBuffer;

  /** Light direction uniform buffer */
  private lightUniformBuffer: GPUBuffer;

  /** Sphere position and radius uniform buffer */
  private sphereUniformBuffer: GPUBuffer;

  /** Shadow toggle flags uniform buffer */
  private shadowUniformBuffer: GPUBuffer;

  /** Water rendering uniforms (density) */
  private waterUniformBuffer: GPUBuffer;

  /** Pool tile texture for refracted view */
  private tileTexture: GPUTexture;

  /** Sampler for tile texture */
  private tileSampler: GPUSampler;

  /** Skybox cubemap texture for reflections */
  private skyTexture: GPUTexture;

  /** Sampler for skybox texture */
  private skySampler: GPUSampler;

  /** Pool half-extent (XZ) in world units — must match `SceneParams` GPU buffer */
  private poolHalfExtent: number;

  /** Shared with pool / surface / caustics / sphere (`SceneParams` struct) */
  private sceneParamsBuffer: GPUBuffer;

  // --- Physics State ---
  // Double-buffered textures for ping-pong rendering

  /**
   * Primary simulation texture (current state).
   * RGBA channels store:
   * - R: Water height
   * - G: Water velocity
   * - B: Surface normal X component
   * - A: Surface normal Z component
   */
  textureA: GPUTexture;

  /**
   * Secondary simulation texture (next state).
   * Swapped with textureA after each simulation step.
   */
  textureB: GPUTexture;

  /**
   * Caustics texture storing light intensity patterns.
   * Higher resolution (1024x1024) for visual detail.
   * - R: Light intensity
   * - G: Sphere shadow factor
   */
  causticsTexture: GPUTexture;

  /** Sampler for simulation textures (linear filtering, clamp edges) */
  sampler: GPUSampler;

  /** Shared simulation tuning (injection, wave response, velocity damping) for all sim passes */
  private simulationParamsBuffer: GPUBuffer;

  /** Pixel format of `textureA` / `textureB` (must match readback decode). */
  private simTextureFormat: GPUTextureFormat;

  /** Staging buffer for 3×3 sim texel readback (GPU → CPU, async). */
  private readbackStaging!: GPUBuffer;

  /** Last successfully mapped height sample (lags GPU by ~1 frame). */
  private waveCpuSample: WaveCpuSample = { ...EMPTY_WAVE_CPU_SAMPLE };

  /** Prevents overlapping copy/map while a prior readback is still mapping. */
  private readbackInFlight = false;

  // --- Simulation Pipelines ---
  // Each pipeline performs one step of the simulation

  /** Pipeline for adding water drops (ripples) */
  private dropPipeline!: PipelineConfig;

  /** Pipeline for wave propagation physics */
  private updatePipeline!: PipelineConfig;

  /** Pipeline for computing surface normals */
  private normalPipeline!: PipelineConfig;

  /** Pipeline for sphere-water interaction */
  private spherePipeline!: PipelineConfig;

  // --- Surface Rendering ---

  /** Vertex buffer for water surface mesh */
  private positionBuffer!: GPUBuffer;

  /** Index buffer for water surface mesh */
  private indexBuffer!: GPUBuffer;

  /** Number of indices in the surface mesh */
  private vertexCount!: number;

  /** Bind group layout for surface rendering (shared by both pipelines) */
  private surfaceBindGroupLayout!: GPUBindGroupLayout;

  /** Pipeline for rendering water surface from above */
  private surfacePipelineAbove!: GPURenderPipeline;

  /** Pipeline for rendering water surface from below */
  private surfacePipelineUnder!: GPURenderPipeline;

  // --- Caustics ---

  /** Pipeline for rendering caustic light patterns */
  private causticsPipeline!: GPURenderPipeline;

  /**
   * Creates a new Water simulation system.
   *
   * @param device - WebGPU device
   * @param width - Simulation texture width
   * @param height - Simulation texture height
   * @param uniformBuffer - Common uniforms (matrices, eye position)
   * @param lightUniformBuffer - Light direction buffer
   * @param sphereUniformBuffer - Sphere position/radius buffer
   * @param shadowUniformBuffer - Shadow toggle flags buffer
   * @param waterUniformBuffer - Water rendering uniforms buffer
   * @param tileTexture - Pool tile texture
   * @param tileSampler - Tile texture sampler
   * @param skyTexture - Skybox cubemap texture
   * @param skySampler - Skybox sampler
   * @param sceneParamsBuffer - Pool dimensions for GPU shaders
   * @param poolHalfExtent - Initial horizontal half-size (X/Z) in world units
   */
  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    uniformBuffer: GPUBuffer,
    lightUniformBuffer: GPUBuffer,
    sphereUniformBuffer: GPUBuffer,
    shadowUniformBuffer: GPUBuffer,
    waterUniformBuffer: GPUBuffer,
    tileTexture: GPUTexture,
    tileSampler: GPUSampler,
    skyTexture: GPUTexture,
    skySampler: GPUSampler,
    sceneParamsBuffer: GPUBuffer,
    poolHalfExtent: number
  ) {
    this.device = device;
    this.width = width;
    this.height = height;

    // Store external resources
    this.commonUniformBuffer = uniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.shadowUniformBuffer = shadowUniformBuffer;
    this.waterUniformBuffer = waterUniformBuffer;
    this.tileTexture = tileTexture;
    this.tileSampler = tileSampler;
    this.skyTexture = skyTexture;
    this.skySampler = skySampler;
    this.sceneParamsBuffer = sceneParamsBuffer;
    this.poolHalfExtent = poolHalfExtent;

    this.simTextureFormat = this.device.features.has('float32-filterable')
      ? 'rgba32float'
      : 'rgba16float';

    // Create double-buffered simulation textures
    this.textureA = this.createTexture();
    this.textureB = this.createTexture();

    this.readbackStaging = this.device.createBuffer({
      label: 'Water sim height readback 3x3',
      size: 3 * 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Caustics texture (higher resolution for detail)
    this.causticsTexture = this.device.createTexture({
      size: [1024, 1024],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Create sampler with linear filtering and edge clamping
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.simulationParamsBuffer = device.createBuffer({
      label: 'Simulation params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.setSimulationParams(0.1, 1.0, 0.997);

    // Initialize all pipelines
    this.createPipelines();
    this.createSurfaceMesh();
    this.createSurfacePipeline();
    this.createCausticsPipeline();
  }

  /**
   * Creates a simulation texture with appropriate format.
   *
   * Uses float32 if available (higher precision), otherwise float16.
   * The texture stores height, velocity, and normal data in RGBA channels.
   */
  private createTexture(): GPUTexture {
    return this.device.createTexture({
      size: [this.width, this.height],
      format: this.simTextureFormat,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC,
    });
  }

  /**
   * Creates all simulation pipelines (drop, update, normal, sphere).
   *
   * Each pipeline renders a fullscreen quad that processes every pixel
   * of the simulation texture. The output is written to textureB,
   * then textures are swapped.
   */
  private createPipelines(): void {
    const format: GPUTextureFormat = this.simTextureFormat;

    // --- Drop Pipeline ---
    // Adds circular ripples to the water at a given position
    // Uses cosine falloff for smooth drop shape
    this.dropPipeline = this.createPipeline(
      'Drop',
      fullscreenVertShader,
      dropFragShader,
      32,
      format
    );

    // --- Update Pipeline ---
    // Propagates waves using a simple finite difference scheme
    // Height moves toward neighbor average, velocity carries momentum
    this.updatePipeline = this.createPipeline(
      'Update',
      fullscreenVertShader,
      updateFragShader,
      16,
      format
    );

    // --- Normal Pipeline ---
    // Computes surface normals from height differences
    // Normals are stored in BA channels for lighting calculations
    this.normalPipeline = this.createPipeline(
      'Normal',
      fullscreenVertShader,
      normalFragShader,
      16,
      format
    );

    // --- Sphere Interaction Pipeline ---
    // Displaces water based on sphere movement
    // Adds volume where sphere leaves, removes where it enters
    this.spherePipeline = this.createPipeline(
      'Sphere',
      fullscreenVertShader,
      sphereFragShader,
      32,
      format
    );
  }

  /**
   * Helper to create a simulation pipeline.
   *
   * @param label - Debug label for the pipeline
   * @param vsCode - Vertex shader WGSL code
   * @param fsCode - Fragment shader WGSL code
   * @param uniformSize - Size of the uniform buffer in bytes
   * @param format - Texture format for output
   * @returns PipelineConfig with pipeline and uniform buffer
   */
  private createPipeline(
    label: string,
    vsCode: string,
    fsCode: string,
    uniformSize: number,
    format: GPUTextureFormat
  ): PipelineConfig {
    const module = this.device.createShaderModule({
      label: label + ' Module',
      code: vsCode + fsCode,
    });

    const pipeline = this.device.createRenderPipeline({
      label: label + ' Pipeline',
      layout: 'auto',
      vertex: {
        module: module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: module,
        entryPoint: 'fs_main',
        targets: [{ format: format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    return {
      pipeline,
      uniformSize,
      uniformBuffer: this.device.createBuffer({
        size: uniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  /**
   * Executes a simulation pipeline pass.
   *
   * Renders textureA through the pipeline to textureB,
   * then swaps the textures for double-buffering.
   *
   * @param pipelineObj - The pipeline configuration to run
   * @param uniformsData - Uniform data to upload
   */
  private runPipeline(pipelineObj: PipelineConfig, uniformsData: Float32Array<ArrayBuffer>): void {
    // Upload uniforms
    this.device.queue.writeBuffer(pipelineObj.uniformBuffer, 0, uniformsData);

    // Create bind group with input texture and uniforms
    const bindGroup = this.device.createBindGroup({
      layout: pipelineObj.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.textureA.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: pipelineObj.uniformBuffer } },
        { binding: 3, resource: { buffer: this.simulationParamsBuffer } },
        { binding: 4, resource: { buffer: this.sceneParamsBuffer } },
      ],
    });

    // Execute render pass
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.textureB.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });

    pass.setPipeline(pipelineObj.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6); // Fullscreen quad (2 triangles)
    pass.end();

    this.device.queue.submit([encoder.finish()]);

    // Swap textures for double-buffering
    const temp = this.textureA;
    this.textureA = this.textureB;
    this.textureB = temp;
  }

  /**
   * Adds a circular ripple to the water surface.
   *
   * @param x - X position in [-1, 1] range
   * @param y - Y position in [-1, 1] range
   * @param radius - Radius of the ripple
   * @param strength - Intensity (positive = up, negative = down)
   */
  addDrop(x: number, y: number, radius: number, strength: number): void {
    const data = new Float32Array(4);
    data[0] = x;
    data[1] = y;
    data[2] = radius;
    data[3] = strength;
    this.runPipeline(this.dropPipeline, data);
  }

  /**
   * Advances the water simulation by one time step.
   *
   * Should be called multiple times per frame for smoother simulation.
   */
  stepSimulation(): void {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.updatePipeline, data);
  }

  /**
   * Updates GPU uniforms for wave simulation feel (sphere displacement strength,
   * per-substep Laplacian coupling, per-substep velocity damping).
   * `waveResponse` / `damping` are per-substep values uploaded to the update shader (CPU applies the
   * legacy 2×/frame impulse split across substeps; no frame-dt multiplier).
   */
  setSimulationParams(sphereInject: number, waveResponse: number, damping: number): void {
    const data = new Float32Array([sphereInject, waveResponse, damping, 0]);
    this.device.queue.writeBuffer(this.simulationParamsBuffer, 0, data);
  }

  /**
   * Recomputes surface normals from current height data.
   *
   * Should be called after simulation steps, before rendering.
   */
  updateNormals(): void {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.normalPipeline, data);
  }

  /**
   * Updates water displacement based on sphere movement.
   *
   * @param oldCenter - Previous sphere position [x, y, z]
   * @param newCenter - Current sphere position [x, y, z]
   * @param radius - Sphere radius
   */
  moveSphere(oldCenter: number[], newCenter: number[], radius: number): void {
    const data = new Float32Array(8);
    data[0] = oldCenter[0];
    data[1] = oldCenter[1];
    data[2] = oldCenter[2];
    data[3] = radius;
    data[4] = newCenter[0];
    data[5] = newCenter[1];
    data[6] = newCenter[2];
    data[7] = 0; // padding
    this.runPipeline(this.spherePipeline, data);
  }

  // =========================================================================
  // Surface Rendering
  // =========================================================================

  /**
   * Creates the water surface mesh as a subdivided plane.
   *
   * The plane spans [-poolHalfExtent, poolHalfExtent] on X and Z axes.
   * Higher detail (200x200) provides smooth displacement from wave heights.
   */
  private createSurfaceMesh(): void {
    const detail = 200; // Grid resolution
    const positions: number[] = [];
    const indices: number[] = [];

    // Generate vertex grid covering the pool in X and Z (world units)
    for (let z = 0; z <= detail; z++) {
      const t = z / detail;
      for (let x = 0; x <= detail; x++) {
        const s = x / detail;
        // Store as XY initially (Z will be sampled from texture)
        positions.push((2 * s - 1) * this.poolHalfExtent, (2 * t - 1) * this.poolHalfExtent, 0);
      }
    }

    // Generate triangle indices
    for (let z = 0; z < detail; z++) {
      for (let x = 0; x < detail; x++) {
        const i = x + z * (detail + 1);
        // Two triangles per quad
        indices.push(i, i + 1, i + detail + 1);
        indices.push(i + detail + 1, i + 1, i + detail + 2);
      }
    }

    this.vertexCount = indices.length;

    // Create vertex buffer
    this.positionBuffer = this.device.createBuffer({
      label: 'Water Surface Vertices',
      size: positions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
    this.positionBuffer.unmap();

    // Create index buffer
    this.indexBuffer = this.device.createBuffer({
      label: 'Water Surface Indices',
      size: indices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }

  /**
   * Rebuilds the water surface mesh when pool horizontal size changes.
   */
  rebuildSurfaceMesh(halfExtent: number): void {
    this.poolHalfExtent = halfExtent;
    this.createSurfaceMesh();
  }

  /**
   * Creates the water surface rendering pipelines.
   *
   * Two pipelines are created:
   * - Above: For viewing water from above (culls front faces)
   * - Under: For viewing water from below (culls back faces)
   *
   * The shader implements ray tracing for reflections and refractions,
   * with Fresnel blending between them.
   */
  private createSurfacePipeline(): void {
    /**
     * Creates vertex shader module.
     */
    const createVertexShaderModule = (label: string, vertCode: string): GPUShaderModule => {
      return this.device.createShaderModule({
        label: `${label} Vertex Shader`,
        code: vertCode,
      });
    };

    /**
     * Creates fragment shader module.
     */
    const createFragmentShaderModule = (label: string, fragCode: string): GPUShaderModule => {
      return this.device.createShaderModule({
        label: `${label} Fragment Shader`,
        code: fragCode,
      });
    };

    // Create bind group layout (shared by both pipelines)
    this.surfaceBindGroupLayout = this.device.createBindGroupLayout({
      label: 'Water Surface BindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        {
          binding: 12,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });

    const surfacePipelineLayout = this.device.createPipelineLayout({
      label: 'Water Surface PipelineLayout',
      bindGroupLayouts: [this.surfaceBindGroupLayout],
    });

    /**
     * Helper to create a surface pipeline with specific settings.
     */
    const createSurfacePipeline = (
      label: string,
      vertShader: string,
      fragShader: string,
      cullMode: GPUCullMode
    ): GPURenderPipeline => {
      const vertexShaderModule = createVertexShaderModule(label, vertShader);
      const fragmentShaderModule = createFragmentShaderModule(label, fragShader);

      return this.device.createRenderPipeline({
        label,
        layout: surfacePipelineLayout,
        vertex: {
          module: vertexShaderModule,
          entryPoint: 'vs_main',
          buffers: [
            {
              arrayStride: 3 * 4,
              attributes: [
                {
                  shaderLocation: 0,
                  offset: 0,
                  format: 'float32x3',
                },
              ],
            },
          ],
        },
        fragment: {
          module: fragmentShaderModule,
          entryPoint: 'fs_main',
          targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode,
        },
        depthStencil: {
          depthWriteEnabled: true,
          depthCompare: 'less',
          format: 'depth24plus',
        },
      });
    };

    // Create both pipelines
    this.surfacePipelineAbove = createSurfacePipeline(
      'Water Surface Above Pipeline',
      surfaceVertShader,
      surfaceAboveFragShader,
      'front' // Cull front faces (see back face = top of water)
    );
    this.surfacePipelineUnder = createSurfacePipeline(
      'Water Surface Under Pipeline',
      surfaceVertShader,
      surfaceUnderFragShader,
      'back' // Cull back faces (see front face = bottom of water)
    );
  }

  /**
   * Renders the water surface to the current render pass.
   *
   * Renders twice: once for above-water view, once for underwater view.
   * The appropriate pipeline is selected based on face culling.
   *
   * @param passEncoder - The active render pass encoder
   */
  renderSurface(passEncoder: GPURenderPassEncoder): void {
    // Create bind group with all required resources
    const bindGroup = this.device.createBindGroup({
      layout: this.surfaceBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.commonUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 3, resource: this.tileSampler },
        { binding: 4, resource: this.tileTexture.createView() },
        { binding: 5, resource: this.sampler },
        { binding: 6, resource: this.textureA.createView() },
        { binding: 7, resource: this.skySampler },
        { binding: 8, resource: this.skyTexture.createView({ dimension: 'cube' }) },
        { binding: 9, resource: this.causticsTexture.createView() },
        { binding: 10, resource: { buffer: this.shadowUniformBuffer } },
        { binding: 11, resource: { buffer: this.waterUniformBuffer } },
        { binding: 12, resource: { buffer: this.sceneParamsBuffer } },
      ],
    });

    // Render water surface from above
    passEncoder.setPipeline(this.surfacePipelineAbove);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);

    // Render water surface from below (same geometry, different shader)
    passEncoder.setPipeline(this.surfacePipelineUnder);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.drawIndexed(this.vertexCount);
  }

  // =========================================================================
  // Caustics Rendering
  // =========================================================================

  /**
   * Creates the caustics rendering pipeline.
   *
   * Caustics are the light patterns on the pool floor caused by
   * refraction through the water surface. The algorithm:
   * 1. For each water surface vertex, trace refracted light ray to pool floor
   * 2. Compare old position (flat water) to new position (displaced water)
   * 3. Light intensity is proportional to area ratio (convergence = brighter)
   *
   * Uses additive blending to accumulate light from multiple rays.
   */
  private createCausticsPipeline(): void {
    // Create separate shader modules for vertex and fragment stages
    const vertexShaderModule = this.device.createShaderModule({
      label: 'Caustics Vertex Shader',
      code: causticsVertShader,
    });

    const fragmentShaderModule = this.device.createShaderModule({
      label: 'Caustics Fragment Shader',
      code: causticsFragShader,
    });

    this.causticsPipeline = this.device.createRenderPipeline({
      label: 'Caustics Pipeline',
      layout: 'auto',
      vertex: {
        module: vertexShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 3 * 4,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
            ],
          },
        ],
      },
      fragment: {
        module: fragmentShaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: 'rgba8unorm',
            // Additive blending: multiple rays contribute to same pixel
            blend: {
              color: {
                operation: 'add',
                srcFactor: 'one',
                dstFactor: 'one',
              },
              alpha: {
                operation: 'add',
                srcFactor: 'one',
                dstFactor: 'one',
              },
            },
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  /**
   * Updates the caustics texture.
   *
   * Should be called after water simulation and normal updates.
   * The caustics texture is then used by pool and sphere shaders.
   */
  updateCaustics(): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.causticsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.lightUniformBuffer } },
        { binding: 1, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.textureA.createView() },
        { binding: 4, resource: { buffer: this.shadowUniformBuffer } },
        { binding: 5, resource: { buffer: this.waterUniformBuffer } },
        { binding: 6, resource: { buffer: this.sceneParamsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.causticsTexture.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });

    pass.setPipeline(this.causticsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, this.positionBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.vertexCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /** Minimum simulation UV step — use for foam / derivative sampling in shaders. */
  getWaterTexel(): number {
    return Math.min(1 / this.width, 1 / this.height);
  }

  /**
   * Latest completed async readback (may be stale by 1–2 frames). Used for CPU wave–body coupling.
   */
  getWaveCpuSample(): WaveCpuSample {
    return this.waveCpuSample;
  }

  /**
   * Copies a 3×3 region of sim height (texture R) around the world-XZ position into a staging buffer
   * and updates `getWaveCpuSample()` when the map completes. Safe to call every frame; skips if a copy
   * is already in flight.
   */
  queueWaveHeightReadback(worldX: number, worldZ: number): void {
    if (this.readbackInFlight) {
      return;
    }
    this.readbackInFlight = true;
    const poolHalf = this.poolHalfExtent;
    const uvx = worldX * (0.5 / poolHalf) + 0.5;
    const uvy = worldZ * (0.5 / poolHalf) + 0.5;
    const ix = Math.min(this.width - 2, Math.max(1, Math.floor(uvx * this.width)));
    const iy = Math.min(this.height - 2, Math.max(1, Math.floor(uvy * this.height)));
    const originX = ix - 1;
    const originY = iy - 1;

    const bytesPerRow = 256;
    const encoder = this.device.createCommandEncoder({ label: 'Water sim height readback' });
    encoder.copyTextureToBuffer(
      { texture: this.textureA, origin: { x: originX, y: originY, z: 0 } },
      { buffer: this.readbackStaging, offset: 0, bytesPerRow },
      { width: 3, height: 3, depthOrArrayLayers: 1 }
    );
    this.device.queue.submit([encoder.finish()]);

    const format = this.simTextureFormat;
    const staging = this.readbackStaging;

    staging
      .mapAsync(GPUMapMode.READ, 0, 3 * bytesPerRow)
      .then(() => {
        const mapped = staging.getMappedRange();
        const ab = mapped.slice(0);
        staging.unmap();
        this.readbackInFlight = false;

        const pixelBytes = format === 'rgba32float' ? 16 : 8;
        const readR = (row: number, col: number): number => {
          const byteOffset = row * bytesPerRow + col * pixelBytes;
          if (format === 'rgba32float') {
            return new Float32Array(ab, byteOffset, 1)[0];
          }
          return new DataView(ab, byteOffset, 2).getFloat16(0, true);
        };

        const h10 = readR(1, 0);
        const h11 = readR(1, 1);
        const h12 = readR(1, 2);
        const h01 = readR(0, 1);
        const h21 = readR(2, 1);

        const worldDx = (2 * poolHalf) / this.width;
        const worldDz = (2 * poolHalf) / this.height;
        const dSimDx = (h12 - h10) / (2 * worldDx);
        const dSimDz = (h21 - h01) / (2 * worldDz);
        const scale = poolHalf / REF_POOL_HALF_EXTENT;
        const dhdxWorld = scale * dSimDx;
        const dhdzWorld = scale * dSimDz;
        const simH = h11;
        const surfaceWorldY = waterHeightWorld(simH, poolHalf);

        this.waveCpuSample = {
          valid: true,
          simH,
          surfaceWorldY,
          dhdxWorld,
          dhdzWorld,
        };
      })
      .catch(() => {
        this.readbackInFlight = false;
      });
  }

  /**
   * Updates the water rendering uniform buffer.
   *
   * @param density - Water density (absorption coefficient)
   * @param causticIntensity - Intensity of caustics
   * @param ior - Index of refraction
   * @param fresnelMin - Minimum fresnel reflection (artistic floor)
   * @param surfaceRoughness - Environment reflection blur / sun softness (0–1)
   * @param foamStrength - Laplacian foam amount (0–1)
   * @param waterTexel - min(1/w,1/h) for height-field derivatives
   */
  updateWaterParameters(
    density: number,
    causticIntensity: number,
    ior: number,
    fresnelMin: number,
    surfaceRoughness: number,
    foamStrength: number,
    waterTexel: number
  ): void {
    this.device.queue.writeBuffer(
      this.waterUniformBuffer,
      0,
      new Float32Array([
        density,
        causticIntensity,
        ior,
        fresnelMin,
        surfaceRoughness,
        foamStrength,
        waterTexel,
        0,
      ])
    );
  }
}
