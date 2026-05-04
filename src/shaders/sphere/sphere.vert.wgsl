#include "../common/bindings.wgsl"

// Bindings
@binding(0) @group(0) var<uniform> commonUniforms : CommonUniforms;
@binding(1) @group(0) var<uniform> sphereUniforms : SphereUniforms;
@binding(2) @group(0) var<uniform> light : LightUniforms;
@binding(3) @group(0) var waterSampler : sampler;
@binding(4) @group(0) var waterTexture : texture_2d<f32>;
@binding(5) @group(0) var causticTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) modelNormal : vec3f, // unit model-space normal (rotated with mesh)
  @location(1) worldPos : vec3f,
}

fn rotateY(v : vec3f, ang : f32) -> vec3f {
  let c = cos(ang);
  let s = sin(ang);
  return vec3f(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
}

fn rotateX(v : vec3f, ang : f32) -> vec3f {
  let c = cos(ang);
  let s = sin(ang);
  return vec3f(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

fn rotateZ(v : vec3f, ang : f32) -> vec3f {
  let c = cos(ang);
  let s = sin(ang);
  return vec3f(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}

@vertex
fn vs_main(@location(0) position : vec3f, @location(1) normal : vec3f) -> VertexOutput {
  var output : VertexOutput;

  // Ry(spinY) → Rx(wavePitch) → Rz(waveRoll): UFO spin, then align hull with local wave slope.
  var p = rotateY(position, sphereUniforms.spinY);
  var n = rotateY(normalize(normal), sphereUniforms.spinY);
  p = rotateX(p, sphereUniforms.wavePitch);
  n = rotateX(n, sphereUniforms.wavePitch);
  p = rotateZ(p, sphereUniforms.waveRoll);
  n = rotateZ(n, sphereUniforms.waveRoll);

  let worldPos = sphereUniforms.center + p * sphereUniforms.radius;
  output.position = commonUniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);
  output.modelNormal = normalize(n);
  output.worldPos = worldPos;
  return output;
}
