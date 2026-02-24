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
struct DemoBindlessExtendedMaterialIndices {
  // The index of the `DemoBindlessExtendedMaterial` data in
  // `example_extended_material`.
  material: u32,
}

// Plain data associated with this demo material.
struct DemoBindlessExtendedMaterial {
  modulate_color: vec4<f32>,
}  

#ifdef BINDLESS
// The indices of the bindless resources in the bindless resource arrays, for
// the `DemoBindlessExtension` fields.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<storage> demo_extended_material_indices: array<DemoBindlessExtendedMaterialIndices>;

// An array that holds the `DemoBindlessExtendedMaterial` plain old data,
// indexed by `DemoBindlessExtendedMaterialIndices.material`.
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage> demo_extended_material: array<DemoBindlessExtendedMaterial>;

#else

// In non-bindless mode, we simply use a uniform for the plain old data.
@group(#{MATERIAL_BIND_GROUP}) @binding(50) var<uniform> demo_extended_material: DemoBindlessExtendedMaterial;

#endif

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifdef BINDLESS
    // Fetch the material slot. We'll use this in turn to fetch the bindless
    // indices from `demo_extended_material_indices`.
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
#endif

    // Generate a `PbrInput` struct from the `StandardMaterial` bindings.
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Calculate the UV for the texture we're about to sample.
#ifdef BINDLESS
    let uv_transform = material_array[material_indices[slot].material].uv_transform;
#else
    let uv_transform = material.uv_transform;
#endif
    let uv = (uv_transform * vec3(in.uv, 1.0)).xy;

    // Multiply the base color by the `modular_texture` and `modulate_color`.
#ifdef BINDLESS
    // Notice how we fetch the texture, sampler, and plain extended material
    // data from the appropriate arrays.
    pbr_input.material.base_color *= demo_extended_material[demo_extended_material_indices[slot].material].modulate_color;
#else
    pbr_input.material.base_color *= demo_extended_material.modulate_color;
#endif

    var out: FragmentOutput;
    // Apply lighting.
    out.color = apply_pbr_lighting(pbr_input);
    // Apply in-shader post processing (fog, alpha-premultiply, and also
    // tonemapping, debanding if the camera is non-HDR). Note this does not
    // include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
} 
