#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

@group(0) @binding(0) var waterTexture : texture_2d<f32>;
@group(0) @binding(1) var waterSampler : sampler;

struct NormalUniforms {
  delta : vec2f,  // Texel size (1/width, 1/height)
}
@group(0) @binding(2) var<uniform> u : NormalUniforms;

struct SimParams {
  sphereInject : f32,
  waveResponse : f32,
  damping : f32,
  _pad : f32,
}
@group(0) @binding(3) var<uniform> sim : SimParams;
@group(0) @binding(4) var<uniform> scene : SceneParams;

@fragment
fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
  var info = textureSample(waterTexture, waterSampler, uv);

  // Sample neighboring heights (mirror at pool rim so edge normals match reflected wave sim)
  let val_dx = textureSample(waterTexture, waterSampler, reflectPoolUv(vec2f(uv.x + u.delta.x, uv.y))).r;
  let val_dy = textureSample(waterTexture, waterSampler, reflectPoolUv(vec2f(uv.x, uv.y + u.delta.y))).r;

  // World position: x,z = (uv - 0.5) * 2 * POOL_XZ_HALF, so ∂x/∂u = 2*POOL_XZ_HALF (world units per UV)
  let dWorld : f32 = 2.0 * scene.poolHalfExtent;
  let hScale : f32 = scene.poolHalfExtent / REF_POOL_HALF_EXTENT;
  let dx = vec3f(u.delta.x * dWorld, hScale * (val_dx - info.r), 0.0);
  let dy = vec3f(0.0, hScale * (val_dy - info.r), u.delta.y * dWorld);

  // Normal is cross product of tangent vectors
  let normal = normalize(cross(dy, dx));
  info.b = normal.x + sim.sphereInject * 0.0;  // Store X component (sim: shared bind layout)
  info.a = normal.z + sim.waveResponse * 0.0;  // Store Z component

  return info;
}
