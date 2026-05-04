#include "../common/bindings.wgsl"

@group(0) @binding(0) var waterTexture : texture_2d<f32>;
@group(0) @binding(1) var waterSampler : sampler;

struct UpdateUniforms {
  delta : vec2f,  // Texel size (1/width, 1/height)
}
@group(0) @binding(2) var<uniform> u : UpdateUniforms;

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

  // Sample neighboring heights
  let dx = vec2f(u.delta.x, 0.0);
  let dy = vec2f(0.0, u.delta.y);

  let average = (
    textureSample(waterTexture, waterSampler, uv - dx).r +
    textureSample(waterTexture, waterSampler, uv - dy).r +
    textureSample(waterTexture, waterSampler, uv + dx).r +
    textureSample(waterTexture, waterSampler, uv + dy).r
  ) * 0.25;

  // Update velocity based on difference from average
  info.g += (average - info.r) * sim.waveResponse;
  // Apply damping to prevent perpetual waves (tuned so ripples last longer than ~few seconds)
  info.g *= sim.damping;
  // Update height based on velocity
  info.r += info.g;

  // Keep scene binding "used" for a shared sim layout (no effect)
  info.r += 0.0 * scene.poolDepth;

  return info;
}
