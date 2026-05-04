#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

// Bindings (structs are in common/bindings.wgsl)
@binding(0) @group(0) var<uniform> commonUniforms : CommonUniforms;
@binding(1) @group(0) var<uniform> sphereUniforms : SphereUniforms;
@binding(2) @group(0) var<uniform> light : LightUniforms;
@binding(3) @group(0) var waterSampler : sampler;
@binding(4) @group(0) var waterTexture : texture_2d<f32>;
@binding(5) @group(0) var causticTexture : texture_2d<f32>;
@binding(6) @group(0) var<uniform> scene : SceneParams;

// Seamless triplanar grain (no UV unwrap). `offsetFromCenter` must be object-local (e.g. worldPos - center),
// not raw world position, or the pattern slides when the object is translated.
fn triplanarGrain(offsetFromCenter : vec3f, n : vec3f, scale : f32) -> f32 {
  let p = offsetFromCenter * scale;
  let w = abs(n);
  let ws = w.x + w.y + w.z + 1e-5;
  let wx = w.x / ws;
  let wy = w.y / ws;
  let wz = w.z / ws;
  let gx = sin(p.y * 2.17 + p.z * 1.83) * cos(p.x * 1.41 + p.y * 0.61);
  let gy = sin(p.x * 1.91 + p.z * 2.07) * cos(p.y * 1.33 + p.z * 0.73);
  let gz = sin(p.x * 2.03 + p.y * 1.77) * cos(p.z * 1.51 + p.x * 0.89);
  return gx * wx + gy * wy + gz * wz;
}

// Three uncorrelated scalar grains for a cheap micro-normal (catches light).
fn triplanarGrainBumped(offsetFromCenter : vec3f, n : vec3f, scale : f32) -> vec3f {
  return vec3f(
    triplanarGrain(offsetFromCenter, n, scale),
    triplanarGrain(offsetFromCenter + vec3f(41.7, 13.2, 7.9), n, scale * 1.02),
    triplanarGrain(offsetFromCenter + vec3f(9.1, 52.3, 21.4), n, scale * 0.98)
  );
}

@fragment
fn fs_main(@location(0) modelNormal : vec3f, @location(1) worldPos : vec3f) -> @location(0) vec4f {
  // Physical constants for light refraction
  let IOR_AIR = 1.0;
  let IOR_WATER = 1.333;

  // Base color: gray sphere, slight metallic teal for UFO
  var color = select(vec3f(0.5), vec3f(0.32, 0.48, 0.44), sphereUniforms.shapeKind > 0.5);

  // Grain in center-relative space so it is glued to the mesh when the object moves (translation).
  let grainPos = worldPos - sphereUniforms.center;

  let sphereNormal = normalize(modelNormal);
  let isUfo = sphereUniforms.shapeKind > 0.5;
  let scCoarse = select(11.0, 13.5, isUfo);
  let scFine = select(38.0, 46.0, isUfo);
  let coarse = triplanarGrain(grainPos, sphereNormal, scCoarse);
  let fine = triplanarGrain(grainPos, sphereNormal, scFine);
  let grain = coarse * 0.65 + fine * 0.35;
  let grain01 = grain * 0.5 + 0.5;
  // Push contrast so speckle survives pool tint and underwater fog.
  let grainVis = pow(clamp(grain01, 0.02, 0.98), 0.82);
  color *= mix(vec3f(0.52), vec3f(1.28), grainVis);
  // Micro-bump so highlights/scales vary across the surface (reads much stronger than albedo alone).
  let bumpAmt = select(0.14, 0.11, isUfo);
  let bumpVec = triplanarGrainBumped(grainPos, sphereNormal, scFine * 0.48);
  let nLit = normalize(sphereNormal + bumpVec * bumpAmt);

  let sphereRadius = sphereUniforms.radius;
  let point = worldPos;
  let xzH = scene.poolHalfExtent;
  let xzUv = 0.5 / xzH;

  // Distance-based darkening near pool boundaries
  // Creates ambient occlusion effect near walls and floor
  let dist_x = (xzH + sphereRadius - abs(point.x)) / sphereRadius;
  let dist_z = (xzH + sphereRadius - abs(point.z)) / sphereRadius;
  let dist_y = (point.y + scene.poolDepth + sphereRadius) / sphereRadius;

  // Apply inverse-cube falloff for soft shadows
  color *= 1.0 - 0.9 / pow(max(0.1, dist_x), 3.0);
  color *= 1.0 - 0.9 / pow(max(0.1, dist_z), 3.0);
  color *= 1.0 - 0.9 / pow(max(0.1, dist_y), 3.0);

  // Calculate refracted light direction (Snell's law)
  let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

  // Basic diffuse lighting (bumped normal makes grain visible in motion and at grazing angles).
  var diffuse = max(0.0, dot(-refractedLight, nLit)) * 0.5;
  diffuse *= mix(0.78, 1.22, grain01);

  // Sample water height at sphere's XZ position
  let waterInfo = textureSampleLevel(waterTexture, waterSampler, poolXZToUv(point.xz, xzH), 0.0);

  // Apply caustics when underwater
  if (point.y < waterHeightWorld(waterInfo.r, xzH)) {
     // Project caustic UV based on refracted light direction
     let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * xzUv + vec2f(0.5);
     let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
     diffuse *= caustic.r * 4.0; // Amplify caustic brightness
  }

  color += diffuse;

  if (point.y < waterHeightWorld(waterInfo.r, xzH)) {
     color *= scene.underTint.rgb * 1.2;
     let viewPath = length(point - commonUniforms.eyePosition);
     color *= beerLambertTransmittance(viewPath, scene.waterAbsorption);
  }

  return vec4f(color, 1.0);
}
