// Eyelash texture baking.
//
// Mirrors the fragment-shader assignment in `hs2_head_eyelashes_material.wgsl`
// (non-bindless branch): the sampled DXT texture's R channel carries the
// eyelash alpha mask, so we bake
//
//   base_color = vec4(0.0, color.g, color.b, color.r)
//
// into the output texture. Lighting / post-processing are NOT baked; the
// output is later sampled as `StandardMaterial::base_color_texture`.

@group(0) @binding(60) var eyelash_texture: texture_2d<f32>;
@group(0) @binding(61) var eyelash_sampler: sampler;

@group(0) @binding(62) var output: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let out_size = vec2<f32>(textureDimensions(output));
  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let uv = (vec2<f32>(invocation_id.xy) + vec2(0.5)) / out_size;

  let eyelash_color = textureSampleLevel(eyelash_texture, eyelash_sampler, uv, 0.0);

  let final_color = vec4<f32>(0.0, eyelash_color.g, eyelash_color.b, eyelash_color.r);

  textureStore(output, location, final_color);
}
