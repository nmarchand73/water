#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

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

fn sampleHeightMirrored(sampleUv : vec2f) -> f32 {
  return textureSample(waterTexture, waterSampler, reflectPoolUv(sampleUv)).r;
}

@fragment
fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
  var info = textureSample(waterTexture, waterSampler, uv);

  let dx = vec2f(u.delta.x, 0.0);
  let dy = vec2f(0.0, u.delta.y);

  // Classic stable wave coupling: neighbor average minus height (same as original demo).
  // Neighbors use rim reflection — isotropic 9-point Laplacian was removed (much stiffer → blow-ups).
  let average = (
    sampleHeightMirrored(uv - dx) +
    sampleHeightMirrored(uv - dy) +
    sampleHeightMirrored(uv + dx) +
    sampleHeightMirrored(uv + dy)
  ) * 0.25;

  // Clamp curvature drive — stops runaway resonance while keeping ripples (original demo had no dt scaling).
  let dh = clamp(average - info.r, -MAX_WAVE_CURVATURE_STEP, MAX_WAVE_CURVATURE_STEP);

  // sim.waveResponse / sim.damping are per-substep values (CPU; no frame-dt multiplier).
  info.g += dh * sim.waveResponse;
  info.g *= sim.damping;
  // Clamp velocity before integrating height — reduces single-frame spikes when neighbor clamps lag.
  info.g = clamp(info.g, -MAX_WATER_SIM_VELOCITY, MAX_WATER_SIM_VELOCITY);
  info.r += info.g;

  info.r += 0.0 * scene.poolDepth;

  return clampWaterSimRg(info);
}
