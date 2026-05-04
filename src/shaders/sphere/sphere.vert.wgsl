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

@vertex
fn vs_main(@location(0) position : vec3f, @location(1) normal : vec3f) -> VertexOutput {
  var output : VertexOutput;

  let c = cos(sphereUniforms.spinY);
  let s = sin(sphereUniforms.spinY);
  let rx = position.x * c - position.z * s;
  let rz = position.x * s + position.z * c;
  let p = vec3f(rx, position.y, rz);
  let nx = normal.x * c - normal.z * s;
  let nz = normal.x * s + normal.z * c;
  let n = normalize(vec3f(nx, normal.y, nz));

  let worldPos = sphereUniforms.center + p * sphereUniforms.radius;
  output.position = commonUniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);
  output.modelNormal = n;
  output.worldPos = worldPos;
  return output;
}
