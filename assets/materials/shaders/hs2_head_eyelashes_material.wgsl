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
struct EyelashesMaterialtIndices {
  // The index of the `EyeMaterialExt` data in
  // `example_extended_material`.
  material: u32,

  // The index of the texture we're going to modulate the base color with in
  // the `bindless_textures_2d` array.
  eyelash_texture: u32,
  // The index of the sampler we're going to sample the modulated texture with
  // in the `bindless_samplers_filtering` array.
  eyelash_sampler: u32,

}
