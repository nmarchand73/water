#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

// Bindings (structs are in common/bindings.wgsl)
@binding(0) @group(0) var<uniform> light : LightUniforms;
@binding(1) @group(0) var<uniform> sphere : SphereUniforms;
@binding(4) @group(0) var<uniform> shadows : ShadowUniforms;
@binding(6) @group(0) var<uniform> scene : SceneParams;

// Water simulation texture
@binding(2) @group(0) var waterSampler : sampler;
@binding(3) @group(0) var waterTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) oldPos : vec3f,  // Where ray would hit with flat water
  @location(1) newPos : vec3f,  // Where ray hits with displaced water
  @location(2) ray : vec3f,     // Refracted ray direction
}

// Projects ray from water surface to pool floor
fn project(origin: vec3f, ray: vec3f, refractedLight: vec3f) -> vec3f {
    let poolHeight = scene.poolDepth;
    let h = scene.poolHalfExtent;
    var point = origin;

    // First find where ray exits pool volume
    let tcube = intersectCube(origin, ray, vec3f(-h, -poolHeight, -h), vec3f(h, scene.poolRimMaxY, h));
    point += ray * tcube.y;

    // Then project down to floor plane (y = -poolDepth)
    let tplane = (-point.y - poolHeight) / refractedLight.y;
    return point + refractedLight * tplane;
}

@vertex
fn vs_main(@location(0) position : vec3f) -> VertexOutput {
  var output : VertexOutput;
  let uv = poolXZToUv(position.xy, scene.poolHalfExtent);

  // Sample water height and normal
  let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);

  // Reconstruct normal (scaled down for stability)
  let ba = info.ba * 0.5;
  let normal = vec3f(ba.x, sqrt(max(0.0, 1.0 - dot(ba, ba))), ba.y);

  // Calculate refracted light directions
  let IOR_AIR = 1.0;
  let IOR_WATER = 1.333;
  let lightDir = normalize(light.direction);

  // Flat water refraction (reference)
  let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
  // Displaced water refraction (actual)
  let ray = refract(-lightDir, normal, IOR_AIR / IOR_WATER);

  // Water surface position
  let pos = vec3f(position.x, 0.0, position.y);

  // Project both rays to pool floor
  output.oldPos = project(pos, refractedLight, refractedLight);
  output.newPos = project(pos + vec3f(0.0, waterHeightWorld(info.r, scene.poolHalfExtent), 0.0), ray, refractedLight);
  output.ray = ray;

  // Position in caustics texture space (scale so wider pools match original NDC coverage)
  let floorUvScale = 1.0 / scene.poolHalfExtent;
  let projectedPos = 0.75 * (output.newPos.xz - output.newPos.y * refractedLight.xz / refractedLight.y) * floorUvScale;
  output.position = vec4f(projectedPos.x, -projectedPos.y, 0.0, 1.0);

  return output;
}
