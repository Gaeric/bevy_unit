// Bakes the HS2 hair material template (`test_hair` in the prefab).
//
// The template is a plain Principled BSDF, not a skin shader:
//   * Base Color  = constant (the importer writes the character's hair color)
//   * Alpha       = texture alpha
//   * Metallic    = texture green channel
//   * Roughness   = 0.4364 (constant)
//   * Normal      = *_converted normal map, recentered per channel
//
// The Blender-only Displacement (two Wave Textures) is not ported: it is a
// render-time strand effect, not part of the game material.

struct HairBakeParams {
    // rgb = hair color (linear).
    hair_color: vec4<f32>,
    // x = roughness, y = gamma of the normal map red channel,
    // z = gamma of the normal map green channel.
    surface: vec4<f32>,
}

@group(0) @binding(0)
var hair_texture: texture_2d<f32>;
@group(0) @binding(1)
var hair_sampler: sampler;
@group(0) @binding(2)
var bump_texture: texture_2d<f32>;
@group(0) @binding(3)
var bump_sampler: sampler;
@group(0) @binding(4)
var<uniform> params: HairBakeParams;

@group(0) @binding(5)
var out_base_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6)
var out_normal: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7)
var out_metallic_roughness: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(out_base_color));
    if any(invocation_id.xy >= size) {
        return;
    }

    let uv = (vec2<f32>(invocation_id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);

    let texel = textureSampleLevel(hair_texture, hair_sampler, uv, 0.0);
    let normal_texel = textureSampleLevel(bump_texture, bump_sampler, uv, 0.0);

    // Same recentering as the skin bake: the *_converted maps do not have their
    // medians at 0.5, so a per-channel gamma is applied before decoding.
    let recentered = pow(max(normal_texel.rg, vec2<f32>(0.0)), params.surface.yz);
    let normal = normalize(vec3<f32>((recentered - 0.5) * 2.0, 1.0));

    let pixel = vec2<i32>(invocation_id.xy);
    textureStore(out_base_color, pixel, vec4<f32>(params.hair_color.rgb, texel.a));
    textureStore(out_normal, pixel, vec4<f32>(normal * 0.5 + 0.5, 1.0));
    // Metallic comes from the texture's green channel, roughness from the
    // material constant.
    textureStore(out_metallic_roughness, pixel, vec4<f32>(1.0, params.surface.x, texel.g, 1.0));
}
