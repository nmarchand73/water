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

// Sphere for ray intersection and displacement
struct SphereUniforms {
  center : vec3f,
  radius : f32,
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
