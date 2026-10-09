// The shader reference `example-manual-material`.
// This code demonstrates how to write shaders that are compatible with both
// bindless and non-bindless mode. See the `@if(BINDLESS)` blocks.

import bevy_pbr::render::{
    forward_io::{FragmentOutput, VertexOutput},
    mesh_bindings::mesh,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};
import bevy_render::color_operations::{hsv_to_rgb, rgb_to_hsv};
import bevy_render::bindless::{bindless_samplers_filtering, bindless_textures_2d};

@if(BINDLESS)
import bevy_pbr::render::pbr_bindings::{material_array, material_indices};
@else
import bevy_pbr::render::pbr_bindings::material;

// Stores the indices of the bindless resources in the bindless resource arrays,
// for the `HeadMaterialExt` fields.
struct HeadMaterialExtIndices {
  material: u32,
  main_texture: u32,
  main_sampler: u32,
}

// Plain data associated with this material.
struct HeadMaterialExt {
  color: vec4<f32>,
}

@if(BINDLESS) {
// The indices of the bindless resources in the bindless resource arrays, for
// the `HeadMaterialExt` fields.
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var<storage> head_material_ext_indices: array<HeadMaterialExtIndices>;

// An array that holds the `HeadMaterialExt` plain old data,
// indexed by `HeadMaterialExtIndices.material`.
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var<storage> head_material_ext: array<HeadMaterialExt>;
}
// In non-bindless mode, we simply use a uniform for the plain old data.
@else {
@group(constants::MATERIAL_BIND_GROUP) @binding(80) var<uniform> head_material_ext: HeadMaterialExt;
@group(constants::MATERIAL_BIND_GROUP) @binding(81) var main_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(82) var main_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(83) var detail_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(84) var detail_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(85) var detail_gloss_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(86) var detail_gloss_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(87) var eyebrow_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(88) var eyebrow_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(89) var bump_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(90) var bump_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(91) var bump_ex_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(92) var bump_ex_sampler: sampler;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
  // Fetch the material slot. We'll use this in turn to fetch the bindless
  // indices from `head_material_ext_indices`.
  @if(BINDLESS)
  let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;

  // Calculate the UV for the texture we're about to sample.
  @if(BINDLESS)
  let uv_transform = material_array[material_indices[slot].material].uv_transform;
  @else
  let uv_transform = material.uv_transform;

  var pbr_input = pbr_input_from_standard_material(in, is_front);
  let uv = (uv_transform * vec3(in.uv, 1.0)).xy;

  @if(BINDLESS)
  let main_color = textureSample(
      bindless_textures_2d[head_material_ext_indices[slot].main_texture],
      bindless_samplers_filtering[head_material_ext_indices[slot].main_sampler],
      uv,
    );
  @else
  let main_color = textureSample(main_texture, main_sampler, uv);

  // very simple tone mapping
  var hsv_color = rgb_to_hsv(main_color.xyz);
  hsv_color.z = sqrt(sqrt(hsv_color.z));
  let base_color = vec4(hsv_to_rgb(hsv_color), main_color.a);

  pbr_input.material.base_color = base_color;

  var out: FragmentOutput;
  out.color = apply_pbr_lighting(pbr_input);
  out.color = main_pass_post_lighting_processing(pbr_input, out.color);

  return out;
}
