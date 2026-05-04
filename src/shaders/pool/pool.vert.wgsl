#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

// Bindings
@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var tileSampler : sampler;
@binding(2) @group(0) var tileTexture : texture_2d<f32>;
@binding(3) @group(0) var<uniform> light : LightUniforms;
@binding(4) @group(0) var<uniform> sphere : SphereUniforms;
@binding(5) @group(0) var waterSampler : sampler;
@binding(6) @group(0) var waterTexture : texture_2d<f32>;
@binding(7) @group(0) var causticTexture : texture_2d<f32>;
@binding(8) @group(0) var<uniform> shadows : ShadowUniforms;
@binding(9) @group(0) var<uniform> scene : SceneParams;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) localPos : vec3f,
}

@vertex
fn vs_main(@location(0) position : vec3f) -> VertexOutput {
  var output : VertexOutput;

  // Transform Y coordinate to create pool depth (scale default profile by poolDepth)
  var transformedPos = position;
  transformedPos.y = ((1.0 - position.y) * (7.0 / 12.0) - 1.0) * scene.poolDepth;

  output.position = uniforms.modelViewProjectionMatrix * vec4f(transformedPos, 1.0);
  output.localPos = transformedPos;
  return output;
}
