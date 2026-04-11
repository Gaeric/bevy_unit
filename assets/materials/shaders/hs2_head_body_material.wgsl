// The shader reference `example-manual-material`.
// This code demonstrates how to write shaders that are compatible with both
// bindless and non-bindless mode. See the `#ifdef BINDLESS` blocks.

#import bevy_pbr::{
    forward_io::{FragmentOutput, VertexOutput},
    mesh_bindings::mesh,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}

#import bevy_render::{color_operations::{hsv_to_rgb, rgb_to_hsv}}

#import bevy_render::bindless::{bindless_samplers_filtering, bindless_textures_2d}

#ifdef BINDLESS
#import bevy_pbr::pbr_bindings::{material_array, material_indices}
#else
#import bevy_pbr::pbr_bindings::material
#endif

struct BodyMaterialExtIndices {
  main_texture: u32,
  main_sampler: u32,
}


#ifdef BINDLESS
// The indices of the bindless resources in the bindless resource arrays, for
// the `DemoBindlessExtension` fields.
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<storage> body_material_ext_indices: array<BodyMaterialExtIndices>;

#else

// In non-bindless mode, we simply use a uniform for the plain old data.
@group(#{MATERIAL_BIND_GROUP}) @binding(200) var main_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(201) var main_sampler: sampler;

#endif


@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifdef BINDLESS
  let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
#endif
  
#ifdef BINDLESS
  let uv_transform = material_array[material_indices[slot].material].uv_transform;
#else
  let uv_transform = material.uv_transform;
#endif

  var pbr_input = pbr_input_from_standard_material(in, is_front);
  let uv = (uv_transform * vec3(in.uv, 1.0)).xy;


#ifdef BINDLESS
  let main_color = textureSample(bindless_textures_2d[body_material_ext_indices[slot].main_texture],
                                 bindless_samplers_filtering[body_material_ext_indices[slot].main_sampler],
                                 uv);
#else
  let main_color = textureSample(main_texture, main_sampler, uv);
#endif

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
