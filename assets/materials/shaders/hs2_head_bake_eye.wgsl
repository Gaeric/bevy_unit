// Eye texture baking.
//
// Mirrors the fragment-shader color math in `hs2_head_eye_material.wgsl`:
// sclera / iris / highlight / pupil are sampled and combined into a single
// base-color map. `iris_color` is a per-material parameter passed as a
// uniform. Lighting / post-processing are NOT baked; the output is later
// sampled as `StandardMaterial::base_color_texture`.

struct EyeBakeParams {
  iris_color: vec4<f32>,
}

@group(0) @binding(60) var sclera_texture: texture_2d<f32>;
@group(0) @binding(61) var sclera_sampler: sampler;
@group(0) @binding(62) var iris_texture: texture_2d<f32>;
@group(0) @binding(63) var iris_sampler: sampler;
@group(0) @binding(64) var highlight_texture: texture_2d<f32>;
@group(0) @binding(65) var highlight_sampler: sampler;
@group(0) @binding(66) var pupil_texture: texture_2d<f32>;
@group(0) @binding(67) var pupil_sampler: sampler;

@group(0) @binding(68) var output: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(69) var<uniform> params: EyeBakeParams;

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let out_size = vec2<f32>(textureDimensions(output));
  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let uv = (vec2<f32>(invocation_id.xy) + vec2(0.5)) / out_size;

  let sclera_color = textureSampleLevel(sclera_texture, sclera_sampler, uv, 0.0);
  let iris_color = textureSampleLevel(iris_texture, iris_sampler, uv, 0.0);
  let highlight_color = textureSampleLevel(highlight_texture, highlight_sampler, uv, 0.0);

  // todo: use scale from uniform
  let pupil_uv = (uv - 0.5) * 3.7 + 0.5;
  let pupil_color = textureSampleLevel(pupil_texture, pupil_sampler, pupil_uv, 0.0);

  let rec_709_coeffs = vec3<f32>(0.2126, 0.7152, 0.0722);

  let substract_value_a = 1.0;
  let substract_value_b = dot(pupil_color.rgb * pupil_color.a, rec_709_coeffs);
  let substract_result = substract_value_a - substract_value_b;

  let multiply1_result = sclera_color * vec4<f32>(0.84);

  // multiply node
  let multiply_value_a = substract_result;
  let multiply_value_b = iris_color.r;
  let multiply_result = multiply_value_a * multiply_value_b;

  // mix node
  let mix1_factor = multiply_result;
  // black color
  let mix1_color_a = vec4<f32>(0.0);
  let mix1_color_b = params.iris_color;
  let mix1_result = mix(mix1_color_a, mix1_color_b, saturate(mix1_factor));

  // mix node
  let mix_factor = iris_color.b;
  let mix_color_a = multiply1_result;
  let mix_color_b = mix1_result;
  let mix_result = mix(mix_color_a, mix_color_b, saturate(mix_factor));

  // add node
  // todo: uniform params
  let add_factor = 0.5;
  let color_a = mix_result;
  let color_b = highlight_color;
  let add_result = color_a + saturate(add_factor) * color_b;

  let final_color = vec4<f32>(add_result.rgb, 1.0);

  textureStore(output, location, final_color);
}
