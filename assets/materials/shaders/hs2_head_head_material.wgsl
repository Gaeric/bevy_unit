// The shader reference `example-manual-material`.
// This code demonstrates how to write shaders that are compatible with both
// bindless and non-bindless mode. See the `#ifdef BINDLESS` blocks.

#import bevy_pbr::{
    forward_io::{FragmentOutput, VertexOutput},
    mesh_bindings::mesh,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#import bevy_render::bindless::{bindless_samplers_filtering, bindless_textures_2d}

#ifdef BINDLESS
#import bevy_pbr::pbr_bindings::{material_array, material_indices}
#else
#import bevy_pbr::pbr_bindings::material
#endif

struct HeadMaterialExtIndices {
  material: u32,
}

struct HeadMaterialExt {
  color: vec4<f32>
}

#ifdef BINDLESS
// The indices of the bindless resources in the bindless resource arrays, for
// the `DemoBindlessExtension` fields.
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<storage> head_material_ext_indices: array<HeadMaterialExtIndices>;

// An array that holds the `EyeMaterialExt` plain old data,
// indexed by `EyeMaterialExtIndices.material`.
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<storage> head_material_ext: array<HeadMaterialExt>;

#else

// In non-bindless mode, we simply use a uniform for the plain old data.
@group(#{MATERIAL_BIND_GROUP}) @binding(80) var<uniform> head_material_ext: HeadMaterialExt;
@group(#{MATERIAL_BIND_GROUP}) @binding(81) var main_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(82) var main_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(83) var detail_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(84) var detail_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(85) var detail_gloss_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(86) var detail_gloss_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(87) var eyebrow_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(88) var eyebrow_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(89) var bump_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var bump_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(91) var bump_ex_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(92) var bump_ex_sampler: sampler;

#endif


@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifdef BINDLESS
  let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
#else
  let slot = in.material_bind_group_slot;
#endif
  
#ifdef BINDLESS
  let uv_transform = material_array[material_indices[slot].material].uv_transform;
#else
  let uv_transform = material.uv_transform;
#endif

  var pbr_input = pbr_input_from_standard_material(in, is_front);
  let uv = (uv_transform * vec3(in.uv, 1.0)).xy;

  var out: FragmentOutput;
  out.color = apply_pbr_lighting(pbr_input);
  out.color = main_pass_post_lighting_processing(pbr_input, out.color);

  return out;
}
