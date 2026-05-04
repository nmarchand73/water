#include "../common/bindings.wgsl"

@group(0) @binding(0) var waterTexture : texture_2d<f32>;
@group(0) @binding(1) var waterSampler : sampler;

struct DropUniforms {
  center : vec2f,    // Drop position in [-1, 1] range
  radius : f32,      // Drop radius
  strength : f32,    // Drop intensity (positive or negative)
}
@group(0) @binding(2) var<uniform> u : DropUniforms;

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

  // Calculate distance from drop center with cosine falloff
  let drop = max(0.0, 1.0 - length(u.center * 0.5 + 0.5 - uv) / u.radius);
  let dropVal = 0.5 - cos(drop * 3.14159265) * 0.5;

  // Add drop height to water surface (sim binding kept for shared layout with other sim passes)
  info.r += dropVal * u.strength + sim.sphereInject * 0.0 + scene.poolHalfExtent * 0.0;

  return info;
}
