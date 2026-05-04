#include "../common/bindings.wgsl"
#include "../common/functions.wgsl"

// Bindings (structs are in common/bindings.wgsl)
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

// Functions (intersectCube is in common/functions.wgsl)

@fragment
fn fs_main(@location(0) localPos : vec3f) -> @location(0) vec4f {
  var wallColor : vec3f;
  let point = localPos;

  // Sample tile texture based on which face we're rendering
  // Use different coordinate pairs for different wall orientations
  let xzHalf = scene.poolHalfExtent;
  let xzToUv = 0.5 / xzHalf;
  if (abs(point.x) > xzHalf * 0.999) {
    // X-facing walls: use YZ coordinates (horizontal Z uses pool extent)
    wallColor = textureSampleLevel(tileTexture, tileSampler, vec2f(point.y * 0.5 + 1.0, point.z * xzToUv + 0.5), 0.0).rgb;
  } else if (abs(point.z) > xzHalf * 0.999) {
    // Z-facing walls: use YX coordinates
    wallColor = textureSampleLevel(tileTexture, tileSampler, vec2f(point.y * 0.5 + 1.0, point.x * xzToUv + 0.5), 0.0).rgb;
  } else {
    // Floor: use XZ coordinates
    wallColor = textureSampleLevel(tileTexture, tileSampler, poolXZToFloorTileUv(point.xz, xzHalf), 0.0).rgb;
  }

  // Physical constants for light refraction
  let IOR_AIR = 1.0;
  let IOR_WATER = 1.333;
  let poolHeight = scene.poolDepth;

  // Determine surface normal based on face
  var normal = vec3f(0.0, 1.0, 0.0);
  if (abs(point.x) > xzHalf * 0.999) { normal = vec3f(-point.x, 0.0, 0.0); }
  else if (abs(point.z) > xzHalf * 0.999) { normal = vec3f(0.0, 0.0, -point.z); }

    // Ambient occlusion
    var scale = 0.5;
    scale /= length(point);
    scale *= mix(1.0, 1.0 - 0.9 / pow(length(point - sphere.center) / sphere.radius, 4.0), shadows.sphere);

    // Lighting with caustics or rim shadow

  // Calculate refracted light direction (Snell's law)
  let refractedLight = -refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

  // Basic diffuse lighting
  let diffuse = max(0.0, dot(refractedLight, normal));

  // Sample water height at this XZ position
  let waterInfo = textureSampleLevel(waterTexture, waterSampler, poolXZToUv(point.xz, xzHalf), 0.0);

  if (point.y < waterHeightWorld(waterInfo.r, xzHalf)) {
     // UNDERWATER: Apply caustic lighting
     // Project caustic UV based on refracted light direction
     let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * xzToUv + vec2f(0.5);
     let caustic = textureSampleLevel(causticTexture, tileSampler, causticUV, 0.0);

     var intensity = caustic.r;       // Caustic brightness
     var sphereShadow = caustic.g;    // Sphere shadow in caustics

     // Fill black void outside caustic mesh with ambient light when rim shadow is off
     if (shadows.rim < 0.5 && intensity < 0.001) {
         intensity = 0.2;
         sphereShadow = 1.0;
     }

     scale += diffuse * intensity * 2.0 * sphereShadow;
  } else {
     // ABOVE WATER: Apply rim shadow at water edge
     let t = intersectCube(point, refractedLight, vec3f(-xzHalf, -poolHeight, -xzHalf), vec3f(xzHalf, scene.poolRimMaxY, xzHalf));
     let shadowFactor = 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
     scale += diffuse * mix(1.0, shadowFactor, shadows.rim) * 0.5;
  }

  var finalColor = wallColor * scale;

  // Apply underwater color tint
  if (point.y < waterHeightWorld(waterInfo.r, xzHalf)) {
     let underwaterColor = vec3f(0.4, 0.9, 1.0);
     finalColor *= underwaterColor * 1.2;
  }

  return vec4f(finalColor, 1.0);
}
