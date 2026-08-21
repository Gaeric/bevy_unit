
@group(0) @binding(60) var eyelash_texture: texture_2d<f32>;
@group(0) @binding(61) var eyelash_sampler: sampler;

@group(0) @binding(62) var output: texture_storage_2d<rgba32float, read_write>;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let out_size = vec2<f32>(textureDimensions(output));
  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let uv = (vec2<f32>(invocation_id.xy) + vec2(0.5)) / out_size;

  let source_color = textureSampleLevel(eyelash_texture, eyelash_sampler, uv, 0.0);
  let final_color = source_color * vec4(1.0, 0.0, 0.0, 1.0);

  textureStore(output, location, final_color);

  // let random_r = randomFloat((invocation_id.y << 16u) | invocation_id.x);
  // let random_g = randomFloat((invocation_id.y << 16u) | invocation_id.x);
  // let random_b = randomFloat((invocation_id.y << 16u) | invocation_id.x);
  // let random_a = randomFloat((invocation_id.y << 16u) | invocation_id.x);
  // let color = vec4(1.0, random_g, random_b, random_a);
  // textureStore(output, location, color);
}
