/**
 * Minimal Wavefront OBJ parser for triangle meshes with optional per-corner normals.
 * Handles quads (splits into two triangles) and n-gons (fan triangulation).
 * Output is centered and scaled to fit in a unit bounding sphere (max |p| = 1).
 */

export type ObjMesh = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
};

function parseFaceCorner(token: string): { vi: number; vni: number | undefined } {
  const parts = token.trim().split('/');
  const vi = parseInt(parts[0], 10);
  if (!Number.isFinite(vi)) throw new Error(`Bad vertex index in face token: ${token}`);
  let vni: number | undefined;
  if (parts.length >= 3 && parts[2] !== undefined && parts[2] !== '') {
    const n = parseInt(parts[2], 10);
    if (Number.isFinite(n)) vni = n - 1;
  }
  return { vi: vi - 1, vni };
}

function triangulateIndices(faceCorners: number[]): [number, number, number][] {
  const tris: [number, number, number][] = [];
  const n = faceCorners.length;
  if (n < 3) return tris;
  if (n === 3) {
    tris.push([faceCorners[0], faceCorners[1], faceCorners[2]]);
    return tris;
  }
  for (let i = 1; i < n - 1; i++) {
    tris.push([faceCorners[0], faceCorners[i], faceCorners[i + 1]]);
  }
  return tris;
}

function normalizeVec3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

function triangleNormal(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number
): [number, number, number] {
  const abx = bx - ax,
    aby = by - ay,
    abz = bz - az;
  const acx = cx - ax,
    acy = cy - ay,
    acz = cz - az;
  let nx = aby * acz - abz * acy;
  let ny = abz * acx - abx * acz;
  let nz = abx * acy - aby * acx;
  return normalizeVec3(nx, ny, nz);
}

/**
 * Parses OBJ text into indexed position/normal buffers, then fits mesh in unit sphere.
 */
export function parseObjToUnitSphereMesh(objText: string): ObjMesh {
  const vLines: number[] = [];
  const vnLines: number[] = [];

  const lines = objText.split(/\r?\n/);
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('v ') && !t.startsWith('vn ') && !t.startsWith('vt ')) {
      const rest = t.slice(2).trim().split(/\s+/);
      const x = parseFloat(rest[0]);
      const y = parseFloat(rest[1]);
      const z = parseFloat(rest[2]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        vLines.push(x, y, z);
      }
    } else if (t.startsWith('vn ')) {
      const rest = t.slice(3).trim().split(/\s+/);
      const x = parseFloat(rest[0]);
      const y = parseFloat(rest[1]);
      const z = parseFloat(rest[2]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        vnLines.push(x, y, z);
      }
    }
  }

  const vnCount = vnLines.length / 3;
  const vertCount = vLines.length / 3;

  type Corner = { vi: number; vni: number | undefined };
  const faceCornersList: Corner[][] = [];

  for (const line of lines) {
    const t = line.trim();
    if (!t.startsWith('f ')) continue;
    const tokens = t.slice(2).trim().split(/\s+/).filter(Boolean);
    if (tokens.length < 3) continue;
    const corners: Corner[] = [];
    for (const tok of tokens) {
      try {
        corners.push(parseFaceCorner(tok));
      } catch {
        continue;
      }
    }
    if (corners.length >= 3) faceCornersList.push(corners);
  }

  const positionsOut: number[] = [];
  const normalsOut: number[] = [];
  const indicesOut: number[] = [];
  const indexMap = new Map<string, number>();

  function getOrCreateIndex(vi: number, nx: number, ny: number, nz: number): number {
    if (vi < 0 || vi >= vertCount) {
      throw new Error(`OBJ: vertex index out of range: ${vi + 1} (have ${vertCount} positions)`);
    }
    const key = `${vi}|${nx.toFixed(5)},${ny.toFixed(5)},${nz.toFixed(5)}`;
    let idx = indexMap.get(key);
    if (idx !== undefined) return idx;
    idx = positionsOut.length / 3;
    indexMap.set(key, idx);
    const vx = vLines[vi * 3];
    const vy = vLines[vi * 3 + 1];
    const vz = vLines[vi * 3 + 2];
    positionsOut.push(vx, vy, vz);
    normalsOut.push(nx, ny, nz);
    return idx;
  }

  for (const corners of faceCornersList) {
    const resolvedN: { nx: number; ny: number; nz: number }[] = [];
    let missingNormal = false;

    for (const c of corners) {
      if (c.vni !== undefined && c.vni >= 0 && c.vni < vnCount) {
        const nx = vnLines[c.vni * 3];
        const ny = vnLines[c.vni * 3 + 1];
        const nz = vnLines[c.vni * 3 + 2];
        const [nnx, nny, nnz] = normalizeVec3(nx, ny, nz);
        resolvedN.push({ nx: nnx, ny: nny, nz: nnz });
      } else {
        missingNormal = true;
        resolvedN.push({ nx: 0, ny: 0, nz: 0 });
      }
    }

    if (missingNormal) {
      const vi0 = corners[0].vi;
      const vi1 = corners[1].vi;
      const vi2 = corners[2].vi;
      const [fnx, fny, fnz] = triangleNormal(
        vLines[vi0 * 3],
        vLines[vi0 * 3 + 1],
        vLines[vi0 * 3 + 2],
        vLines[vi1 * 3],
        vLines[vi1 * 3 + 1],
        vLines[vi1 * 3 + 2],
        vLines[vi2 * 3],
        vLines[vi2 * 3 + 1],
        vLines[vi2 * 3 + 2]
      );
      for (let i = 0; i < corners.length; i++) {
        if (corners[i].vni === undefined || corners[i].vni! < 0 || corners[i].vni! >= vnCount) {
          resolvedN[i] = { nx: fnx, ny: fny, nz: fnz };
        }
      }
    }

    const idxCorners: number[] = [];
    for (let i = 0; i < corners.length; i++) {
      const c = corners[i];
      const { nx, ny, nz } = resolvedN[i];
      idxCorners.push(getOrCreateIndex(c.vi, nx, ny, nz));
    }

    const tris = triangulateIndices(idxCorners);
    for (const [a, b, c] of tris) {
      indicesOut.push(a, b, c);
    }
  }

  if (positionsOut.length === 0) {
    throw new Error('OBJ: no geometry parsed');
  }

  const positions = new Float32Array(positionsOut);
  const normals = new Float32Array(normalsOut);
  const indices = new Uint32Array(indicesOut);

  // 3ds Max Wavefront export is Z-up (floor in XY, height along +Z). This demo is Y-up
  // (pool floor in XZ, height along +Y). Without this, the saucer sits “on edge” (XY treated as horizontal).
  convertMaxZUpMeshToYUpWorld(positions, normals);

  fitMeshToUnitSphere(positions);

  return { positions, normals, indices };
}

/** Rotation -90° about X: Max (x,y,z) → engine (x, z, -y). Same linear map for normals (renormalized). */
function convertMaxZUpMeshToYUpWorld(positions: Float32Array, normals: Float32Array): void {
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    positions[i * 3] = x;
    positions[i * 3 + 1] = z;
    positions[i * 3 + 2] = -y;
  }
  const m = normals.length / 3;
  for (let i = 0; i < m; i++) {
    const x = normals[i * 3];
    const y = normals[i * 3 + 1];
    const z = normals[i * 3 + 2];
    let nx = x;
    let ny = z;
    let nz = -y;
    const len = Math.hypot(nx, ny, nz) || 1;
    normals[i * 3] = nx / len;
    normals[i * 3 + 1] = ny / len;
    normals[i * 3 + 2] = nz / len;
  }
}

/** Centers mesh at origin and scales so max |p| is 1 (matches physics bounding sphere). */
function fitMeshToUnitSphere(positions: Float32Array): void {
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3],
      y = positions[i * 3 + 1],
      z = positions[i * 3 + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  let maxR = 0;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3] - cx;
    const y = positions[i * 3 + 1] - cy;
    const z = positions[i * 3 + 2] - cz;
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    maxR = Math.max(maxR, Math.hypot(x, y, z));
  }
  if (maxR > 1e-8) {
    for (let i = 0; i < positions.length; i++) {
      positions[i] /= maxR;
    }
  }
}

export async function fetchAndParseUfoObj(url: string): Promise<ObjMesh> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch OBJ: ${res.status} ${url}`);
  const text = await res.text();
  return parseObjToUnitSphereMesh(text);
}
