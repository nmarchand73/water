/**
 * sphere.ts - Interactive Sphere Renderer
 *
 * This module renders a draggable sphere that interacts with the water simulation.
 * The sphere can be moved by the user and optionally affected by physics (gravity
 * and buoyancy).
 *
 * Key features:
 * - Procedurally generated sphere geometry using octahedron subdivision
 * - Caustic lighting effects when underwater
 * - Ambient occlusion based on proximity to pool walls
 * - Underwater color tinting
 */

// Import shader modules
import sphereVertShader from './shaders/sphere/sphere.vert.wgsl';
import sphereFragShader from './shaders/sphere/sphere.frag.wgsl';
import type { ObjMesh } from './shapes/parse-ufo-obj';

/** Revolved “flying saucer” profile (y, r), scaled to unit bounding sphere; spun in the vertex shader. */
function buildUfoRevolvedMesh(): {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
} {
  const raw: [number, number][] = [
    [-0.52, 0.94],
    [-0.3, 0.36],
    [-0.08, 0.56],
    [0.08, 0.46],
    [0.26, 0.3],
    [0.44, 0.08],
  ];
  let maxD = 0;
  for (const [y, r] of raw) {
    maxD = Math.max(maxD, Math.hypot(y, r));
  }
  const profile = raw.map(([y, r]) => [y / maxD, r / maxD] as [number, number]);

  const stacks = profile.length;
  const rings = 40;
  const positions: number[] = [];
  const idxMap: number[][] = [];
  for (let i = 0; i < stacks; i++) {
    idxMap[i] = [];
    const [y, r] = profile[i];
    for (let j = 0; j < rings; j++) {
      const th = (j / rings) * Math.PI * 2;
      const x = r * Math.cos(th);
      const z = r * Math.sin(th);
      idxMap[i][j] = positions.length / 3;
      positions.push(x, y, z);
    }
  }

  const indices: number[] = [];
  for (let i = 0; i < stacks - 1; i++) {
    for (let j = 0; j < rings; j++) {
      const jn = (j + 1) % rings;
      const v00 = idxMap[i][j];
      const v01 = idxMap[i][jn];
      const v10 = idxMap[i + 1][j];
      const v11 = idxMap[i + 1][jn];
      indices.push(v00, v10, v01);
      indices.push(v01, v10, v11);
    }
  }

  const vertCount = positions.length / 3;
  const acc = new Float32Array(vertCount * 3);
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t];
    const ib = indices[t + 1];
    const ic = indices[t + 2];
    const ax = positions[ia * 3],
      ay = positions[ia * 3 + 1],
      az = positions[ia * 3 + 2];
    const bx = positions[ib * 3],
      by = positions[ib * 3 + 1],
      bz = positions[ib * 3 + 2];
    const cx = positions[ic * 3],
      cy = positions[ic * 3 + 1],
      cz = positions[ic * 3 + 2];
    const abx = bx - ax,
      aby = by - ay,
      abz = bz - az;
    const acx = cx - ax,
      acy = cy - ay,
      acz = cz - az;
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    for (const vi of [ia, ib, ic]) {
      acc[vi * 3] += nx;
      acc[vi * 3 + 1] += ny;
      acc[vi * 3 + 2] += nz;
    }
  }
  const normals = new Float32Array(vertCount * 3);
  for (let i = 0; i < vertCount; i++) {
    const x = acc[i * 3],
      y = acc[i * 3 + 1],
      z = acc[i * 3 + 2];
    const L = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / L;
    normals[i * 3 + 1] = y / L;
    normals[i * 3 + 2] = z / L;
  }

  return {
    positions: new Float32Array(positions),
    normals,
    indices: new Uint32Array(indices),
  };
}

/**
 * Renders an interactive sphere with realistic underwater lighting.
 *
 * The Sphere class creates a subdivided octahedron geometry for smooth rendering
 * and applies dynamic lighting based on water state. The shader handles:
 * - Refracted sunlight through water surface
 * - Caustic patterns when underwater
 * - Distance-based darkening near pool edges
 * - Underwater color tinting
 */
export class Sphere {
  /** WebGPU device for creating GPU resources */
  private device: GPUDevice;

  /** Texture format matching the canvas (e.g., 'bgra8unorm') */
  private format: GPUTextureFormat;

  /** Uniform buffer containing view-projection matrix and eye position */
  private commonUniformBuffer: GPUBuffer;

  /** Uniform buffer containing sphere position and radius */
  private sphereUniformBuffer: GPUBuffer;

  /** Uniform buffer containing light direction vector */
  private lightUniformBuffer: GPUBuffer;

  /** Pool dimensions for AO / caustic UV (`SceneParams`) */
  private sceneParamsBuffer: GPUBuffer;

  /** Unit sphere positions + normals (normal = position on unit sphere) */
  private spherePositionBuffer!: GPUBuffer;
  private sphereNormalBuffer!: GPUBuffer;
  private sphereIndexBuffer!: GPUBuffer;
  private sphereIndexCount!: number;

  /** UFO revolved mesh (unit bounding sphere) */
  private ufoPositionBuffer!: GPUBuffer;
  private ufoNormalBuffer!: GPUBuffer;
  private ufoIndexBuffer!: GPUBuffer;
  private ufoIndexCount!: number;

  /** Which mesh to draw */
  private meshKind: 'sphere' | 'ufo' = 'sphere';

  /** The render pipeline for sphere / UFO rendering */
  private pipeline!: GPURenderPipeline;

  /**
   * Creates a new Sphere renderer.
   *
   * @param device - WebGPU device for resource creation
   * @param format - Canvas texture format
   * @param uniformBuffer - Buffer with view-projection matrix and eye position
   * @param lightUniformBuffer - Buffer with light direction
   * @param sphereUniformBuffer - Buffer for sphere position and radius
   * @param sceneParamsBuffer - Pool half-extent / depth uniforms
   */
  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    uniformBuffer: GPUBuffer,
    lightUniformBuffer: GPUBuffer,
    sphereUniformBuffer: GPUBuffer,
    sceneParamsBuffer: GPUBuffer
  ) {
    this.device = device;
    this.format = format;
    this.commonUniformBuffer = uniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;
    this.sceneParamsBuffer = sceneParamsBuffer;

    this.createSphereGeometry();
    this.createUfoGeometry();
    this.createPipeline();
  }

  /**
   * Writes GPU uniforms: center, radius, Y spin (for UFO), shapeKind, wave pitch/roll (radians).
   */
  update(
    center: number[],
    radius: number,
    spinY: number,
    shapeKind: number,
    wavePitch = 0,
    waveRoll = 0
  ): void {
    const data = new Float32Array([
      center[0],
      center[1],
      center[2],
      radius,
      spinY,
      shapeKind,
      wavePitch,
      waveRoll,
    ]);
    this.device.queue.writeBuffer(this.sphereUniformBuffer, 0, data);
  }

  /** Switches rendered mesh between sphere and UFO (same physics bounding sphere). */
  setMeshKind(kind: 'sphere' | 'ufo'): void {
    this.meshKind = kind;
  }

  /**
   * Replace the UFO mesh with external OBJ data (centered, unit bounding sphere — see `parse-ufo-obj`).
   * Destroys the previous UFO GPU buffers (procedural or prior OBJ).
   */
  setUfoMeshFromData(mesh: ObjMesh): void {
    this.ufoPositionBuffer.destroy();
    this.ufoNormalBuffer.destroy();
    this.ufoIndexBuffer.destroy();
    this.ufoIndexCount = mesh.indices.length;

    this.ufoPositionBuffer = this.device.createBuffer({
      label: 'UFO Position Buffer (OBJ)',
      size: mesh.positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.ufoPositionBuffer.getMappedRange()).set(mesh.positions);
    this.ufoPositionBuffer.unmap();

    this.ufoNormalBuffer = this.device.createBuffer({
      label: 'UFO Normal Buffer (OBJ)',
      size: mesh.normals.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.ufoNormalBuffer.getMappedRange()).set(mesh.normals);
    this.ufoNormalBuffer.unmap();

    this.ufoIndexBuffer = this.device.createBuffer({
      label: 'UFO Index Buffer (OBJ)',
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.ufoIndexBuffer.getMappedRange()).set(mesh.indices);
    this.ufoIndexBuffer.unmap();
  }

  /**
   * Creates the sphere geometry using octahedron subdivision.
   *
   * This technique produces a more uniform triangle distribution than
   * latitude/longitude sphere generation. The algorithm:
   * 1. Starts with 8 octants of a unit cube
   * 2. Subdivides each octant into triangles
   * 3. Projects vertices onto a unit sphere
   *
   * The `detail` parameter controls the subdivision level:
   * - detail=1: 8 triangles (octahedron)
   * - detail=10: 800 triangles (smooth sphere)
   */
  private createSphereGeometry(): void {
    const detail = 10; // Subdivision level for smooth sphere

    /**
     * Helper class to deduplicate vertices.
     * Vertices at octant boundaries would be duplicated without this.
     */
    class Indexer {
      /** Array of unique vertex positions */
      unique: number[][];
      /** Map from position string to index */
      map: Map<string, number>;

      constructor() {
        this.unique = [];
        this.map = new Map();
      }

      /**
       * Adds a vertex, returning its index.
       * If the vertex already exists, returns the existing index.
       */
      add(v: number[]): number {
        const key = v.join(',');
        if (!this.map.has(key)) {
          this.map.set(key, this.unique.length);
          this.unique.push(v);
        }
        return this.map.get(key)!;
      }
    }

    /**
     * Returns the sign multipliers for an octant (0-7).
     * Each bit controls the sign of one axis:
     * - Bit 0: X sign
     * - Bit 1: Y sign
     * - Bit 2: Z sign
     */
    function pickOctant(i: number): [number, number, number] {
      return [(i & 1) * 2 - 1, (i & 2) - 1, (i & 4) / 2 - 1];
    }

    /**
     * Applies a smoothing function to make triangles more uniform.
     * Without this, triangles near octant corners would be smaller.
     */
    function fix(x: number): number {
      return x + (x - x * x) / 2;
    }

    const indexer = new Indexer();
    const finalIndices: number[] = [];

    // Process each of the 8 octants
    for (let octant = 0; octant < 8; octant++) {
      const scale = pickOctant(octant);

      // Determine triangle winding based on octant orientation
      const flip = scale[0] * scale[1] * scale[2] > 0;
      const data: number[] = [];

      // Generate vertices for this octant using barycentric subdivision
      for (let i = 0; i <= detail; i++) {
        for (let j = 0; i + j <= detail; j++) {
          // Barycentric coordinates (a, b, c) where a + b + c = 1
          const a = i / detail;
          const b = j / detail;
          const c = (detail - i - j) / detail;

          // Apply smoothing and create position
          const v = [fix(a), fix(b), fix(c)];

          // Normalize to project onto unit sphere
          const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
          const pos = [(v[0] / len) * scale[0], (v[1] / len) * scale[1], (v[2] / len) * scale[2]];

          data.push(indexer.add(pos));
        }
      }

      // Generate triangle indices for this octant
      for (let i = 0; i <= detail; i++) {
        if (i > 0) {
          for (let j = 0; i + j <= detail; j++) {
            // Calculate vertex indices in the triangular grid
            const a = (i - 1) * (detail + 1) + (i - 1 - (i - 1) * (i - 1)) / 2 + j;
            const b = i * (detail + 1) + (i - i * i) / 2 + j;

            // Add triangle with correct winding order
            if (flip) {
              finalIndices.push(data[a], data[b], data[a + 1]);
            } else {
              finalIndices.push(data[a], data[a + 1], data[b]);
            }

            // Add second triangle for quad (except at octant edge)
            if (i + j < detail) {
              if (flip) {
                finalIndices.push(data[b], data[b + 1], data[a + 1]);
              } else {
                finalIndices.push(data[b], data[a + 1], data[b + 1]);
              }
            }
          }
        }
      }
    }

    this.sphereIndexCount = finalIndices.length;

    // Flatten vertex positions; unit sphere: normal = position
    const finalPositions: number[] = [];
    for (const p of indexer.unique) {
      finalPositions.push(...p);
    }
    const finalNormals = new Float32Array(finalPositions);

    this.spherePositionBuffer = this.device.createBuffer({
      label: 'Sphere Position Buffer',
      size: finalPositions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.spherePositionBuffer.getMappedRange()).set(finalPositions);
    this.spherePositionBuffer.unmap();

    this.sphereNormalBuffer = this.device.createBuffer({
      label: 'Sphere Normal Buffer',
      size: finalNormals.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.sphereNormalBuffer.getMappedRange()).set(finalNormals);
    this.sphereNormalBuffer.unmap();

    this.sphereIndexBuffer = this.device.createBuffer({
      label: 'Sphere Index Buffer',
      size: finalIndices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.sphereIndexBuffer.getMappedRange()).set(finalIndices);
    this.sphereIndexBuffer.unmap();
  }

  private createUfoGeometry(): void {
    const { positions, normals, indices } = buildUfoRevolvedMesh();
    this.ufoIndexCount = indices.length;
    this.ufoPositionBuffer = this.device.createBuffer({
      label: 'UFO Position Buffer',
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.ufoPositionBuffer.getMappedRange()).set(positions);
    this.ufoPositionBuffer.unmap();
    this.ufoNormalBuffer = this.device.createBuffer({
      label: 'UFO Normal Buffer',
      size: normals.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.ufoNormalBuffer.getMappedRange()).set(normals);
    this.ufoNormalBuffer.unmap();
    this.ufoIndexBuffer = this.device.createBuffer({
      label: 'UFO Index Buffer',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.ufoIndexBuffer.getMappedRange()).set(indices);
    this.ufoIndexBuffer.unmap();
  }

  /**
   * Creates the render pipeline with WGSL shaders.
   *
   * The shader implements:
   * - Unit sphere scaled and translated by uniform values
   * - Distance-based darkening near pool walls/floor
   * - Refracted light and caustic sampling when underwater
   * - Underwater color tinting
   */
  private createPipeline(): void {
    // Create separate shader modules for vertex and fragment stages
    const vertexShaderModule = this.device.createShaderModule({
      label: 'Sphere Vertex Shader',
      code: sphereVertShader,
    });

    const fragmentShaderModule = this.device.createShaderModule({
      label: 'Sphere Fragment Shader',
      code: sphereFragShader,
    });

    // Create the render pipeline
    this.pipeline = this.device.createRenderPipeline({
      label: 'Sphere Pipeline',
      layout: 'auto',
      vertex: {
        module: vertexShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 3 * 4,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 3 * 4,
            attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
          },
        ],
      },
      fragment: {
        module: fragmentShaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back', // Back-face culling (sphere is closed)
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });
  }

  /**
   * Renders the sphere to the current render pass.
   *
   * Creates a new bind group each frame to incorporate dynamic textures
   * (water height and caustics that change every frame).
   *
   * @param passEncoder - The active render pass encoder
   * @param waterTexture - Current water simulation texture (height/normals)
   * @param waterSampler - Sampler for water texture
   * @param causticsTexture - Pre-computed caustic pattern texture
   */
  render(
    passEncoder: GPURenderPassEncoder,
    waterTexture: GPUTexture,
    waterSampler: GPUSampler,
    causticsTexture: GPUTexture
  ): void {
    // Create bind group with all required resources
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.commonUniformBuffer } },
        { binding: 1, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 2, resource: { buffer: this.lightUniformBuffer } },
        { binding: 3, resource: waterSampler },
        { binding: 4, resource: waterTexture.createView() },
        { binding: 5, resource: causticsTexture.createView() },
        { binding: 6, resource: { buffer: this.sceneParamsBuffer } },
      ],
    });

    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    if (this.meshKind === 'ufo') {
      passEncoder.setVertexBuffer(0, this.ufoPositionBuffer);
      passEncoder.setVertexBuffer(1, this.ufoNormalBuffer);
      passEncoder.setIndexBuffer(this.ufoIndexBuffer, 'uint32');
      passEncoder.drawIndexed(this.ufoIndexCount);
    } else {
      passEncoder.setVertexBuffer(0, this.spherePositionBuffer);
      passEncoder.setVertexBuffer(1, this.sphereNormalBuffer);
      passEncoder.setIndexBuffer(this.sphereIndexBuffer, 'uint32');
      passEncoder.drawIndexed(this.sphereIndexCount);
    }
  }
}
