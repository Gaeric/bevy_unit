// Body (Torso) skin texture baking.
//
// Mirrors the fragment-shader color assignment in
// `hs2_head_body_material.wgsl`: sample the MainTex and apply the simple HSV
// value curve (approximation of the game's texture ramp)
//
//   hsv.z = sqrt(sqrt(hsv.z))
//
// Lighting / post-processing are NOT baked; the output is later sampled as
// `StandardMaterial::base_color_texture`.

#import bevy_render::color_operations::{hsv_to_rgb, rgb_to_hsv}

@group(0) @binding(60) var body_texture: texture_2d<f32>;
@group(0) @binding(61) var body_sampler: sampler;

@group(0) @binding(62) var output: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let out_size = vec2<f32>(textureDimensions(output));
  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let uv = (vec2<f32>(invocation_id.xy) + vec2(0.5)) / out_size;

  let main_color = textureSampleLevel(body_texture, body_sampler, uv, 0.0);

  // very simple tone mapping
  var hsv_color = rgb_to_hsv(main_color.xyz);
  hsv_color.z = sqrt(sqrt(hsv_color.z));
  let final_color = vec4(hsv_to_rgb(hsv_color), main_color.a);

  textureStore(output, location, final_color);
}
