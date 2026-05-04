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

// Sphere / buoy object: center + radius for physics; spin + shape for rendering only
struct SphereUniforms {
  center : vec3f,
  radius : f32,
  spinY : f32,       // radians, applied to UFO mesh (and ignored for sphere shading path)
  shapeKind : f32,   // 0 = sphere, 1 = UFO (same bounding sphere for physics)
  _pad : vec2f,
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
  _pad : f32,
}
