// Eyeshadow texture baking.
//
// Mirrors the fragment-shader assignment in
// `hs2_head_eyeshadow_material.wgsl`: the sampled eyeshadow texture's alpha
// channel carries the mask; the base color is a fixed dark value.
//
//   base_color = vec4(0.016, 0.016, 0.016, eyeshadow_color.a)
//
// Lighting / post-processing are NOT baked; the output is later sampled as
// `StandardMaterial::base_color_texture`.

@group(0) @binding(60) var eyeshadow_texture: texture_2d<f32>;
@group(0) @binding(61) var eyeshadow_sampler: sampler;

@group(0) @binding(62) var output: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let out_size = vec2<f32>(textureDimensions(output));
  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let uv = (vec2<f32>(invocation_id.xy) + vec2(0.5)) / out_size;

  let eyeshadow_color = textureSampleLevel(eyeshadow_texture, eyeshadow_sampler, uv, 0.0);

  let final_color = vec4<f32>(0.016, 0.016, 0.016, eyeshadow_color.a);

  textureStore(output, location, final_color);
}
