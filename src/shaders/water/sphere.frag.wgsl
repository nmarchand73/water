#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

@group(0) @binding(0) var waterTexture : texture_2d<f32>;
@group(0) @binding(1) var waterSampler : sampler;

// Not `SphereUniforms` — that name is the render pass struct in common/bindings.wgsl
struct SphereDisplacementUniforms {
  oldCenter : vec3f,  // Previous sphere position
  radius : f32,       // Sphere radius
  newCenter : vec3f,  // Current sphere position
  padding : f32,      // Alignment padding
}
@group(0) @binding(2) var<uniform> u : SphereDisplacementUniforms;

struct SimParams {
  sphereInject : f32,
  waveResponse : f32,
  damping : f32,
  _pad : f32,
}
@group(0) @binding(3) var<uniform> sim : SimParams;
@group(0) @binding(4) var<uniform> scene : SceneParams;

// Calculates the volume of sphere intersecting the water at a UV position
fn volumeInSphere(center : vec3f, uv : vec2f, radius : f32) -> f32 {
  let h = scene.poolHalfExtent;
  let p = vec3f((uv.x * 2.0 - 1.0) * h, 0.0, (uv.y * 2.0 - 1.0) * h);
  let dist = length(p - center);
  let t = dist / radius;

  // Gaussian-like falloff for smooth interaction
  let dy = exp(-pow(t * 1.5, 6.0));
  let ymin = min(0.0, center.y - dy);
  let ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);
  return (ymax - ymin) * sim.sphereInject;
}

@fragment
fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
  var info = textureSample(waterTexture, waterSampler, uv);

  // Water rises where sphere was, falls where sphere is now
  info.r += volumeInSphere(u.oldCenter, uv, u.radius);
  info.r -= volumeInSphere(u.newCenter, uv, u.radius);

  return info;
}
