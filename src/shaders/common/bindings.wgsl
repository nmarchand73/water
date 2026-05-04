// Common uniform structures and bindings used across multiple shaders

// Common camera uniforms
struct CommonUniforms {
  viewProjectionMatrix : mat4x4f,
  eyePosition : vec3f,
}

// Light direction
struct LightUniforms {
   direction : vec3f,
}

// Sphere / buoy object: center + radius for physics; spin + shape + wave tilt for rendering
struct SphereUniforms {
  center : vec3f,
  radius : f32,
  spinY : f32,       // radians, Y spin (UFO fast; sphere slow — symmetric hull needs motion to read)
  shapeKind : f32,   // 0 = sphere, 1 = UFO (same bounding sphere for physics)
  wavePitch : f32,   // radians, rotation about +X (follows dhdz / swell along Z)
  waveRoll : f32,    // radians, rotation about +Z (follows dhdx / swell along X)
}

// Shadow toggle flags
struct ShadowUniforms {
    rim : f32,      // Rim shadow at water edge
    sphere : f32,   // Sphere ambient occlusion
    ao : f32,       // Pool corner ambient occlusion
}

// Water rendering uniforms
struct WaterUniforms {
    density : f32,
    causticIntensity : f32,
    ior : f32,
    fresnelMin : f32,
    surfaceRoughness : f32,
    foamStrength : f32,
    waterTexel : f32,
    _pad : f32,
}

// Pool uniforms (camera matrices and eye position)
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
  eyePosition : vec3f,
}

// Shared scene scale (must match sceneParamsBuffer in main.ts)
struct SceneParams {
  poolHalfExtent : f32, // horizontal half-size on X and Z (world units)
  poolDepth : f32,      // depth from water plane y=0 to floor (positive)
  poolRimMaxY : f32,    // ray/AABB upper Y for rim shadow (typically >= wall tops)
  waterAbsorption : f32, // Beer-Lambert strength (0 = off, 1 = default extinction)
  underTint : vec4f,    // xyz underwater colorization (w unused)
  tileTint : vec4f,     // xyz multiply pool tile albedo (w unused)
  aboveTint : vec4f,    // xyz volumetric tint above surface (w unused)
}
