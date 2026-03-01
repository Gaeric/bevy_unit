// The shader that goes with `example-manual-material`.
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

// Stores the indices of the bindless resources in the bindless resource arrays,
// for the `DemoBindlessExtension` fields.
struct EyelashesMaterialtExtIndices {
  // The index of the texture we're going to modulate the base color with in
  // the `bindless_textures_2d` array.
  eyelash_texture: u32,
  // The index of the sampler we're going to sample the modulated texture with
  // in the `bindless_samplers_filtering` array.
  eyelash_sampler: u32,
}


#ifdef BINDLESS
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<storage> eyelash_material_ext_indices: array<EyelashesMaterialtExtIndices>;
#else
@group(#{MATERIAL_BIND_GROUP}) @binding(60) var eyelash_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(61) var eyelash_sampler: sampler;
#endif

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifdef BINDLESS
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);

#ifdef BINDLESS
    let uv_transform = material_array[material_indices[slot].material].uv_transform;
#else
    let uv_transform = material.uv_transform;
#endif

    let uv = (uv_transform * vec3(in.uv, 1.0)).xy;

#ifdef BINDLESS
    let eyelash_color = textureSample(bindless_textures_2d[eyelash_material_ext_indices[slot].eyelash_texture],
                                      bindless_samplers_filtering[eyelash_material_ext_indices[slot].eyelash_sampler],
                                      uv);
#else
    let eyelash_color = textureSample(eyelash_texture, eyelash_sampler, uv);
#endif

    pbr_input.material.base_color = vec4<f32>(0.0, eyelash_color.g, eyelash_color.b, eyelash_color.r);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
