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
struct EyeMaterialExtIndices {
  // The index of the `EyeMaterialExt` data in
  // `example_extended_material`.
  material: u32,

  // The index of the texture we're going to modulate the base color with in
  // the `bindless_textures_2d` array.
  sclera_texture: u32,
  // The index of the sampler we're going to sample the modulated texture with
  // in the `bindless_samplers_filtering` array.
  sclera_sampler: u32,

  iris_texture: u32,
  iris_sampler: u32,

  highlight_texture: u32,
  highlight_sampler: u32,

  pupil_texture: u32,
  pupil_sampler: u32,
}

// Plain data associated with this demo material.
struct EyeMaterialExt {
  iris_color: vec4<f32>,
}  

#ifdef BINDLESS
// The indices of the bindless resources in the bindless resource arrays, for
// the `DemoBindlessExtension` fields.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<storage> eye_material_ext_indices: array<EyeMaterialExtIndices>;

// An array that holds the `EyeMaterialExt` plain old data,
// indexed by `EyeMaterialExtIndices.material`.
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage> eye_material_ext: array<EyeMaterialExt>;

#else

// In non-bindless mode, we simply use a uniform for the plain old data.
@group(#{MATERIAL_BIND_GROUP}) @binding(50) var<uniform> eye_material_ext: EyeMaterialExt;
@group(#{MATERIAL_BIND_GROUP}) @binding(51) var sclera_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(52) var sclera_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(53) var iris_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(54) var iris_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(55) var highlight_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(56) var highlight_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(57) var pupil_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(58) var pupil_sampler: sampler;

#endif

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifdef BINDLESS
    // Fetch the material slot. We'll use this in turn to fetch the bindless
    // indices from `eye_material_ext_indices`.
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

    // the uv scale group
    // let corrected_uv = (in.uv - 0.5) * group_input.scale + 0.5;

    // Multiply the base color by the `modular_texture` and `iris_color`.
#ifdef BINDLESS
    // Notice how we fetch the texture, sampler, and plain extended material
    // data from the appropriate arrays.

    let iris_base_color =  eye_material_ext[eye_material_ext_indices[slot].material].iris_color;

    let sclera_color = textureSample(bindless_textures_2d[eye_material_ext_indices[slot].sclera_texture],
                                     bindless_samplers_filtering[eye_material_ext_indices[slot].sclera_sampler],
                                     uv);

    let iris_color = textureSample(bindless_textures_2d[eye_material_ext_indices[slot].iris_texture],
                                     bindless_samplers_filtering[eye_material_ext_indices[slot].iris_sampler],
                                     uv);

    let highlight_color = textureSample(bindless_textures_2d[eye_material_ext_indices[slot].highlight_texture],
                                     bindless_samplers_filtering[eye_material_ext_indices[slot].highlight_sampler],
                                     uv);

    // todo: use scale from uniform
    let pupil_uv = (uv - 0.5) * 3.7 + 0.5;
    let pupil_color = textureSample(bindless_textures_2d[eye_material_ext_indices[slot].pupil_texture],
                                     bindless_samplers_filtering[eye_material_ext_indices[slot].pupil_sampler],
                                     pupil_uv);
#else
    // todo: use scale from uniform
    let pupil_uv = (uv - 0.5) * 3.7 + 0.5;

    let iris_base_color = eye_material_ext.iris_color;
    let sclera_color = textureSample(sclera_texture, sclera_sampler, uv);
    let iris_color = textureSample(iris_texture, iris_sampler, uv);
    let highlight_color = textureSample(highlight_texture, highlight_sampler, uv);
    let pupil_color = textureSample(pupil_texture, pupil_sampler, pupil_uv);
#endif

    let iris_factor = (1.0 - pupil_color) * iris_color.r;
    // let iris_factor = vec4<f32>(1.0);

    pbr_input.material.base_color *= iris_base_color * sclera_color;

    var out: FragmentOutput;
    // Apply lighting.
    out.color = apply_pbr_lighting(pbr_input);
    // Apply in-shader post processing (fog, alpha-premultiply, and also
    // tonemapping, debanding if the camera is non-HDR). Note this does not
    // include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
